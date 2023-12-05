"""Official release of scMoGNN method.

Reference
---------
Wen, Hongzhi, et al. "Graph Neural Networks for Multimodal Single-Cell Data Integration." arXiv:2203.01884 (2022).

"""
import os
from copy import deepcopy

import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.utils.data import DataLoader

from dance import logger
from dance.utils import SimpleIndexDataset
from dance.utils.metrics import *


def propagation_layer_combination(X, idx, wt, from_logits=True):
    if from_logits:
        wt = torch.softmax(wt, -1)

    x = 0
    for i in range(wt.shape[0]):
        x += wt[i] * X[i][idx]

    return x


def cell_feature_propagation(g, alpha: float = 0.5, beta: float = 0.5, cell_init: str = None, feature_init: str = 'id',
                             device: str = 'cuda', layers: int = 3):
    g = g.to(device)
    gconv = dglnn.HeteroGraphConv(
        {
            'cell2feature': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
            'rev_cell2feature': dglnn.GraphConv(in_feats=0, out_feats=0, norm='none', weight=False, bias=False),
        }, aggregate='sum')

    if feature_init is None:
        feature_X = torch.zeros((g.nodes('feature').shape[0], g.srcdata[cell_init]['cell'].shape[1])).float().to(device)
    elif feature_init == 'id':
        feature_X = F.one_hot(g.srcdata['id']['feature']).float().to(device)
    else:
        raise NotImplementedError(f'Not implemented feature init feature {feature_init}.')

    if cell_init is None:
        cell_X = torch.zeros(g.nodes('cell').shape[0], feature_X.shape[1]).float().to(device)
    else:
        cell_X = g.srcdata[cell_init]['cell'].float().to(device)

    h = {'feature': feature_X, 'cell': cell_X}
    hcell = []
    for i in range(layers):
        h1 = gconv(
            g, h, mod_kwargs={
                'cell2feature': {
                    'edge_weight': g.edges['cell2feature'].data['weight'].float()
                },
                'rev_cell2feature': {
                    'edge_weight': g.edges['rev_cell2feature'].data['weight'].float()
                }
            })
        logger.debug(f"{i} cell {h['cell'].abs().mean()} {h1['cell'].abs().mean()}")
        logger.debug(f"{i} feature {h['feature'].abs().mean()} {h1['feature'].abs().mean()}")

        h1['feature'] = (h1['feature'] -
                         h1['feature'].mean()) / (h1['feature'].std() if h1['feature'].mean() != 0 else 1)
        h1['cell'] = (h1['cell'] - h1['cell'].mean()) / (h1['cell'].std() if h1['cell'].mean() != 0 else 1)

        h = {
            'feature': h['feature'] * alpha + h1['feature'] * (1 - alpha),
            'cell': h['cell'] * beta + h1['cell'] * (1 - beta)
        }

        h['feature'] = (h['feature'] - h['feature'].mean()) / h['feature'].std()
        h['cell'] = (h['cell'] - h['cell'].mean()) / h['cell'].std()

        hcell.append(h['cell'])

    logger.debug(f"{hcell[-1].abs().mean()=}")

    return hcell[1:]


class ScMoGCNWrapper:
    """ScMoGCN class.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace object that contains arguments of ScMoGCN. For details of parameters in parser args, please refer
        to link (parser help document).
    dataset : dance.datasets.multimodality.JointEmbeddingNIPSDataset
        Joint embedding dataset.

    """

    def __init__(self, args, num_celL_types, num_batches, num_phases, num_features):
        self.model = ScMoGCN(num_celL_types, num_batches, num_phases, num_features).to(args.device)
        self.args = args
        self.wt = torch.tensor([0.] * (args.layers - 1)).to(args.device).requires_grad_(True)

    def fit(self, g_mod1, g_mod2, train_size, cell_type, batch_label, phase_score):
        """Fit function for training.

        Parameters
        ----------
        g_mod1 : dgl.DGLGraph
            Bipartite expression feature graph for modality 1.
        g_mod2 : dgl.DGLGraph
            Bipartite expression feature graph for modality 2.
        train_size : int
            Number of training samples.
        labels : torch.Tensor
            Labels for training samples.
        cell_type :torch.Tensor
            Cell type labels for training samples.
        batch_label : torch.Tensor
            Batch labels for training samples.
        phase_score : torch.Tensor
            Phase labels for training samples.

        Returns
        -------
        None.

        """

        wt = self.wt
        hcell_mod1 = cell_feature_propagation(g_mod1, layers=self.args.layers, device=self.args.device)
        hcell_mod2 = cell_feature_propagation(g_mod2, layers=self.args.layers, device=self.args.device)
        self.feat_mod1 = hcell_mod1
        self.feat_mod2 = hcell_mod2
        X = []
        for i in range(len(self.feat_mod1)):
            X.append(torch.cat([self.feat_mod1[i], self.feat_mod2[i]], dim=1).float().to(self.args.device))
        self.X = X
        Y = [cell_type.to(self.args.device), batch_label.to(self.args.device), phase_score.float().to(self.args.device)]

        idx = np.random.permutation(train_size)
        train_idx = idx[:int(idx.shape[0] * 0.9)]
        val_idx = idx[int(idx.shape[0] * 0.9):]

        # Make sure the batch size is small enough to cover all splits
        batch_size = min(self.args.batch_size, len(val_idx))

        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
        )

        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        optimizer = torch.optim.AdamW([{'params': self.model.parameters()}, {'params': wt, 'weight_decay': 0}], lr=1e-4)
        vals = []

        for epoch in range(60):
            self.model.train()
            total_loss = [0] * 5
            print('epoch', epoch)

            for iter, batch_idx in enumerate(train_loader):

                batch_x = propagation_layer_combination(X, batch_idx, wt)
                batch_y = [batch_x, Y[0][batch_idx], Y[1][batch_idx], Y[2][batch_idx]]

                output = self.model(batch_x)

                # loss1 = mse(output[0], batch_y[0]) # option 1: recover features after propagation
                loss1 = mse(output[0], batch_x)  # option 2: recover Isi features
                loss2 = ce(output[1], batch_y[1])
                loss3 = torch.norm(output[2], p=2, dim=-1).sum() * 1e-2
                loss4 = mse(output[3], batch_y[3])

                loss = loss1 * 0.7 + loss2 * 0.2 + loss3 * 0.05 + loss4 * 0.05

                total_loss[0] += loss1.item()
                total_loss[1] += loss2.item()
                total_loss[2] += loss3.item()
                total_loss[3] += loss4.item()
                total_loss[4] += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for i in range(4):
                print(f'loss{i + 1}', total_loss[i] / len(train_loader), end=', ')
            print()

            loss1, loss2, loss3, loss4 = self.score(val_idx, Y[0], Y[2])
            weighted_loss = loss1 * 0.7 + loss2 * 0.2 + loss3 * 0.05 + loss4 * 0.05
            print('val-loss1', loss1, 'val-loss2', loss2, 'val-loss3', loss3, 'val-loss4', loss4)
            print('val score', weighted_loss)
            vals.append(weighted_loss)
            if min(vals) == vals[-1]:
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(self.model.state_dict(), f'models/model_joint_embedding_{self.args.seed}.pth')
                weight_record = wt.detach()
                best_dict = deepcopy(self.model.state_dict())

            if min(vals) != min(vals[-10:]):
                print('Early stopped.')
                break

        self.wt = weight_record
        self.fitted = True
        self.model.load_state_dict(best_dict)

    def to(self, device):
        """Performs device conversion.

        Parameters
        ----------
        device : str
            Target device.

        Returns
        -------
        self : ScMoGCNWrapper
            Converted model.

        """

        self.args.device = device
        self.model = self.model.to(device)
        self.feat_mod1 = self.feat_mod1.to(device)
        self.feat_mod2 = self.feat_mod2.to(device)
        self.X = self.X.to(device)
        return self

    def load(self, path, map_location=None):
        """Load model parameters from checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        map_location : str optional
            Mapped device. This parameter will be passed to torch.load function if not none.

        Returns
        -------
        None.

        """
        self.fitted = True
        if map_location is not None:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(path))

    def predict(self, idx):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        idx : Iterable[int]
            Index of testing samples for prediction.

        Returns
        -------
        prediction : torch.Tensor
            Joint embedding of input data.

        """
        if not self.fitted:
            raise RuntimeError('Model is not fitted yet.')
        self.model.eval()
        wt = self.wt
        inputs = self.X

        with torch.no_grad():
            X = propagation_layer_combination(inputs, idx, wt)

            return self.model.encoder(X)

    def score(self, idx, cell_type, phase_score=None, adata_sol=None, metric='loss'):
        """Score function to get score of prediction.

        Parameters
        ----------
        idx : Iterable[int]
            Index of testing samples for scoring.
        cell_type : torch.Tensor
            Cell type labels of testing samples.
        phase_score : torch.Tensor optional
            Cell cycle score of testing samples.
        metric : str optional
            The type of evaluation metric, by default to be 'loss'.

        Returns
        -------
        loss1 : float
            Reconstruction loss.
        loss2 : float
            Cell type classfication loss.
        loss3 : float
            Batch regularization loss.
        loss4 : float
            Cell cycle score loss.

        """

        self.model.eval()
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        inputs = self.X

        with torch.no_grad():
            if metric == 'loss':
                X = propagation_layer_combination(inputs, idx, self.wt)
                output = self.model(X)
                loss1 = mse(output[0], X).item()
                loss2 = ce(output[1], cell_type[idx]).item()
                loss3 = (torch.norm(output[2], p=2, dim=-1).sum() * 1e-2).item()
                loss4 = mse(output[3], phase_score[idx]).item()

                return loss1, loss2, loss3, loss4
            elif metric == 'clustering':
                emb = self.predict(idx).cpu().numpy()
                kmeans = KMeans(n_clusters=10, n_init=5, random_state=200)

                # adata.obs['batch'] = adata_sol.obs['batch'][adata.obs_names]
                # adata.obs['cell_type'] = adata_sol.obs['cell_type'][adata.obs_names]
                true_labels = cell_type
                pred_labels = kmeans.fit_predict(emb)
                NMI_score = round(normalized_mutual_info_score(true_labels, pred_labels, average_method='max'), 3)
                ARI_score = round(adjusted_rand_score(true_labels, pred_labels), 3)

                # print('ARI: ' + str(ARI_score) + ' NMI: ' + str(NMI_score))
                return {'dance_nmi': NMI_score, 'dance_ari': ARI_score}
            elif metric == 'openproblems':
                emb = self.predict(idx).cpu().numpy()
                assert adata_sol, 'adata_sol is required by `openproblems` evaluation but not provided.'
                adata_sol.obsm['X_emb'] = emb
                return integration_openproblems_evaluate(adata_sol)
            else:
                raise NotImplementedError


class ScMoGCN(nn.Module):

    def __init__(self, nb_cell_types, nb_batches, nb_phases, input_dimension):
        super().__init__()
        self.nb_cell_types = nb_cell_types
        self.nb_batches = nb_batches
        self.nb_phases = nb_phases

        self.linear1 = nn.Linear(input_dimension, 150)
        self.linear2 = nn.Linear(150, 120)
        self.linear3 = nn.Linear(120, 100)
        self.linear4 = nn.Linear(100, 61)

        self.bn1 = nn.BatchNorm1d(150)
        self.bn2 = nn.BatchNorm1d(120)
        self.bn3 = nn.BatchNorm1d(100)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.act3 = nn.GELU()

        self.decoder = nn.Sequential(
            nn.Linear(61, 150),
            nn.ReLU(),
            nn.Linear(150, input_dimension),
            nn.ReLU(),
        )

    def encoder(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear3(x)
        x = self.act3(x)
        x = self.bn3(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.linear4(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x0 = x
        x = self.decoder(x)

        return (
            x,
            x0[:, :self.nb_cell_types],
            x0[:, self.nb_cell_types:self.nb_cell_types + self.nb_batches],
            x0[:, self.nb_cell_types + self.nb_batches:self.nb_cell_types + self.nb_batches + self.nb_phases],
        )
