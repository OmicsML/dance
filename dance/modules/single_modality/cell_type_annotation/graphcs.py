import contextlib
import math
import os
import random
import shutil
import tempfile
import time
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dance.modules.base import BaseClassificationMethod
from dance.modules.single_modality.cell_type_annotation.BiPGraph import BiP
from dance.transforms.filter import HighlyVariableGenesLogarithmizedByTopGenes, SupervisedFeatureSelection
from dance.transforms.graph.graphcs import BBKNNConstruction
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import NormalizeTotalLog1P
from dance.typing import LogLevel, Optional


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

class GnnBP(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, bias):
        super(GnnBP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x


class Gnn(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(Gnn, self).__init__()

        self.feature_layers = nn.Sequential(
            nn.Linear(nfeat, 128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(128, nclass))

    def forward(self, x, is_dec = False):
        enc = self.feature_layers(x)
        return enc

def muticlass_f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

# Helper class for Data Loading
class SimpleSet(Data.Dataset):
    def __init__(self, features, labels, names=None):
        self.features = features
        self.labels = labels
        self.names = names if names is not None else np.arange(len(labels))
        self.size = len(labels)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Returns: feature, (label, name)
        return self.features[idx], (self.labels[idx], self.names[idx])


class GraphCSClassifier(BaseClassificationMethod):
    """The GnnBP/GraphCS cell-type classification model.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace contains arguments.
    prj_path: str
        project path for saving temporary checkpoints.
    random_state: int
        Random seed.
    """

    def __init__(self, args, prj_path="./", random_state: Optional[int] = 20159):
        self.prj_path = prj_path
        self.random_state = random_state if random_state is not None else args.seed
        
        # 1. Unpack hyperparameters from args during initialization
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.vat_lr = args.vat_lr
        self.patience = args.patience
        self.gpus = args.gpus
        
        # Model architecture params
        self.layer = args.layer
        self.hidden = args.hidden
        self.dropout = args.dropout
        self.bias = args.bias

        # Setup device
        self.device = torch.device(
            f"cuda:{self.gpus[0]}" if torch.cuda.is_available() and self.gpus else "cpu"
        )
        
        self._set_seed(self.random_state)
        self.model = None

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    @staticmethod
    def preprocessing_pipeline(edge_ratio: float = 2, log_level: LogLevel = "INFO"):
        transforms = []
        transforms.append(SupervisedFeatureSelection(label_col="cell_type", n_features=2000,split_name="train"))
        transforms.append(NormalizeTotalLog1P())
        transforms.append(HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=2000))
        transforms.append(BBKNNConstruction(edge_ratio=edge_ratio, key_added="temp_graph"))
        transforms.append(SetConfig({
            "label_channel": "cell_type"
        }))
        return Compose(*transforms, log_level=log_level)
       

    def fit(self, x_train,y_train,x_val,y_val,nfeat,nclass):
        """Train the GraphCS model."""

        train_dataset = SimpleSet(x_train, y_train)
        
        # Use self.batch_size instead of args.batch
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        # 2. Initialize Model
        # Import GnnBP and VATLoss here to avoid circular imports if they are in utils
        # from model import GnnBP 
        # from vat import VATLoss
        
        
        self.model = GnnBP(
            nfeat=nfeat,
            nlayers=self.layer,
            nhidden=self.hidden,
            nclass=nclass,
            dropout=self.dropout,
            bias=self.bias
        ).to(self.device)

        if len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        # 3. Training Loop
        bad_counter = 0
        best_f1 = 0
        
        if not os.path.exists(os.path.join(self.prj_path, 'pretrained')):
             os.makedirs(os.path.join(self.prj_path, 'pretrained'), exist_ok=True)
        checkpt_file = os.path.join(self.prj_path, 'pretrained', uuid.uuid4().hex + '.pt')

        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            loss_list = []
            
            for batch_x, (batch_y, _) in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss_train = loss_fn(output, batch_y)

                # VAT Loss
                if self.vat_lr > 0:
                    vat_loss_func = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds = vat_loss_func(self.model, batch_x)
                    loss_train += lds * self.vat_lr

                loss_train.backward()
                optimizer.step()
                loss_list.append(loss_train.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_x_gpu = x_val.to(self.device)
                val_y_gpu = y_val.to(self.device)
                output_val = self.model(val_x_gpu)
                # Using a wrapper for f1 metric
                micro_val = muticlass_f1(output_val, val_y_gpu).item()

            avg_loss = np.mean(loss_list)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch:{epoch+1:04d} | loss:{avg_loss:.3f} | val_f1:{micro_val:.6f}')

            # Early Stopping
            if micro_val > best_f1:
                best_f1 = micro_val
                torch.save(self.model.state_dict(), checkpt_file)
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Training finished in {time.time() - start_time:.2f}s")

        # 4. Load Best Model
        self.model.load_state_dict(torch.load(checkpt_file))
        # Optional: os.remove(checkpt_file)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict cell labels."""
        self.model.eval()
        x_tensor = torch.FloatTensor(x)
        
        # Use self.batch_size
        pred_dataset = SimpleSet(x_tensor, torch.zeros(len(x_tensor))) 
        pred_loader = Data.DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, _ in pred_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                pred_batch = output.max(1)[1].cpu().numpy()
                preds.append(pred_batch)
        
        return np.concatenate(preds)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict cell label probabilities."""
        self.model.eval()
        x_tensor = torch.FloatTensor(x)
        pred_dataset = SimpleSet(x_tensor, torch.zeros(len(x_tensor)))
        pred_loader = Data.DataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for batch_x, _ in pred_loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                prob_batch = torch.softmax(output, dim=1).cpu().numpy()
                probs.append(prob_batch)
                
        return np.concatenate(probs, axis=0)
    
    

def load_GBP_data(datastr, alpha, rmax, rrz,temp_data, temp_graph):
    from dance.settings import EXAMPLESDIR
    # 如果提供了临时数据，使用临时文件
    if temp_data is not None and temp_graph is not None:
        os.makedirs("data",exist_ok=True)
        # 保存临时特征文件
        feat_path = os.path.join("data", datastr + "_feat.npy")
        np.save(feat_path, temp_data)

        # 保存临时图文件
        graph_path = os.path.join("data", datastr + ".txt")
        with open(graph_path, 'w') as f:
            for line in temp_graph:
                f.write(line + '\n')

        try:
            features = BiP.ppr(datastr, alpha, rmax, rrz)  #rmax
            features = torch.FloatTensor(features).T
        finally:
            # 清理临时文件
            if os.path.exists(feat_path):
                os.remove(feat_path)
            if os.path.exists(graph_path):
                os.remove(graph_path)
        return features
  
