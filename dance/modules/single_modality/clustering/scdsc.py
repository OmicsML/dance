"""Reimplementation of scDSC.

Extended from https://github.com/DHUDBlab/scDSC

Reference
----------
Gan, Yanglan, et al. "Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph
neural network." Briefings in Bioinformatics 23.2 (2022): bbac018.

"""

import math

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from dance.transforms import AnnDataTransform, Compose, SaveRaw, SetConfig
from dance.transforms.graph import NeighborGraph
from dance.transforms.preprocess import sparse_mx_to_torch_sparse_tensor
from dance.typing import LogLevel
from dance.utils import get_device
from dance.utils.loss import ZINBLoss
from dance.utils.metrics import cluster_acc


class ScDSC:
    """scDSC wrapper class.

    Parameters
    ----------
    pretrain_path
        Path of saved autoencoder weights.
    sigma
        Balance parameter.
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.
    n_clusters
        Number of clusters.
    n_input
        Input feature dimension.
    v
        Parameter of soft assignment.
    device
        Computing device.

    """

    def __init__(
        self,
        pretrain_path: str,
        sigma: float = 1,
        n_enc_1: int = 512,
        n_enc_2: int = 256,
        n_enc_3: int = 256,
        n_dec_1: int = 256,
        n_dec_2: int = 256,
        n_dec_3: int = 512,
        n_z1: int = 256,
        n_z2: int = 128,
        n_z3: int = 32,
        n_clusters: int = 100,
        n_input: int = 10,
        v: float = 1,
        device: str = "auto",
    ):
        super().__init__()
        self.device = get_device(device)
        self.model = ScDSCModel(
            pretrain_path=pretrain_path,
            sigma=sigma,
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
            n_clusters=n_clusters,
            n_input=n_input,
            v=v,
            device=self.device,
        ).to(self.device)
        self.model_pre = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
        ).to(self.device)

    @staticmethod
    def preprocessing_pipeline(n_top_genes: int = 2000, n_neighbors: int = 50, log_level: LogLevel = "INFO"):
        return Compose(
            # Filter data
            AnnDataTransform(sc.pp.filter_genes, min_counts=3),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            AnnDataTransform(sc.pp.normalize_per_cell),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.highly_variable_genes, min_mean=0.0125, max_mean=4, flavor="cell_ranger",
                             min_disp=0.5, n_top_genes=n_top_genes, subset=True),
            # Normalize data
            AnnDataTransform(sc.pp.filter_genes, min_counts=1),
            AnnDataTransform(sc.pp.filter_cells, min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.normalize_total),
            AnnDataTransform(sc.pp.log1p),
            AnnDataTransform(sc.pp.scale),
            # Construct k-neighbors graph using the noramlized feature matrix
            NeighborGraph(n_neighbors=n_neighbors, metric="correlation", channel="X"),
            SetConfig({
                "feature_channel": [None, None, "n_counts", "NeighborGraph"],
                "feature_channel_type": ["X", "raw_X", "obs", "obsp"],
                "label_channel": "Group"
            }),
            log_level=log_level,
        )

    def target_distribution(self, q):
        """Calculate auxiliary target distribution p with q.

        Parameters
        ----------
        q
            Soft label.

        Returns
        -------
        p
            Target distribution.

        """
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def pretrain_ae(self, x, batch_size, n_epochs, fname, lr=1e-3):
        """Pretrain autoencoder.

        Parameters
        ----------
        x
            Input features.
        batch_size
            Size of batch.
        n_epochs
            Number of epochs.
        lr
            Learning rate.
        fname
            Path to save autoencoder weights.

        """
        print("Pretrain:")
        device = self.device
        x_tensor = torch.from_numpy(x)
        train_loader = DataLoader(TensorDataset(x_tensor), batch_size, shuffle=True)
        model = self.model_pre
        optimizer = Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):

            total_loss = total_size = 0
            for batch_idx, (x_batch, ) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                x_bar, _, _, _, _, _, _, _ = model(x_batch)

                loss = F.mse_loss(x_bar, x_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                size = x_batch.shape[0]
                total_size += size
                total_loss += loss.item() * size

            print(f"Pretrain epoch {epoch + 1:4d}, MSE loss:{total_loss / total_size:.8f}")

        torch.save(model.state_dict(), fname)

    def fit(self, x, y, X_raw, n_counts, adj, lr=1e-03, n_epochs=300, bcl=0.1, cl=0.01, rl=1, zl=0.1):
        """Train model.

        Parameters
        ----------
        x
            Input features.
        y
            Labels.
        X_raw
            Raw input features.
        n_counts
            Total counts for each cell.
        adj
            Adjacency matrix as a sicpy sparse matrix.
        lr
            Learning rate.
        n_epochs
            Number of epochs.
        bcl
            Parameter of binary crossentropy loss.
        cl
            Parameter of Kullback–Leibler divergence loss.
        rl
            Parameter of reconstruction loss.
        zl
            Parameter of ZINB loss.

        """
        print("Train:")
        device = self.device
        model = self.model
        optimizer = Adam(model.parameters(), lr=lr)
        # optimizer = RAdam(model.parameters(), lr=lr)

        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        X_raw = torch.tensor(X_raw).to(device)
        sf = torch.tensor(n_counts / np.median(n_counts)).to(device)
        data = torch.from_numpy(x).to(device)

        aris = []
        keys = []
        P = {}
        Q = {}

        with torch.no_grad():
            _, _, _, _, z, _, _, _ = model.ae(data)

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    x_bar, tmp_q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)
                    tmp_q = tmp_q.data
                    self.q = tmp_q
                    p = self.target_distribution(tmp_q)

                    # calculate ari score for model selection
                    _, _, ari = self.score(y)
                    aris.append(ari)
                    keys.append(key := f"epoch{epoch}")
                    print("Epoch %3d, ARI: %.4f, Best ARI: %.4f" % (epoch + 1, ari, max(aris)))

                    P[key] = p
                    Q[key] = tmp_q

            model.train()
            x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss = model(data, adj)

            binary_crossentropy_loss = F.binary_cross_entropy(q, p)
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)
            zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)
            loss = bcl * binary_crossentropy_loss + cl * ce_loss + rl * re_loss + zl * zinb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        index = np.argmax(aris)
        self.q = Q[keys[index]]

    def predict(self):
        """Get predictions from the trained model.

        Returns
        -------
        y_pred
            Prediction of given clustering method.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        return y_pred

    def score(self, y):
        """Evaluate the trained model.

        Parameters
        ----------
        y
            True labels.

        Returns
        -------
        acc
            Accuracy.
        nmi
            Normalized mutual information.
        ari
            Adjusted Rand index.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        return acc, nmi, ari


class ScDSCModel(nn.Module):
    """scDSC class.

    Parameters
    ----------
    pretrain_path
        Path of saved autoencoder weights.
    sigma
        Balance parameter.
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.
    n_clusters
        Number of clusters.
    n_input
        Input feature dimension.
    v
        Parameter of soft assignment.
    device
        Computing device.

    """

    def __init__(
        self,
        pretrain_path: str,
        sigma: float = 1,
        n_enc_1: int = 512,
        n_enc_2: int = 256,
        n_enc_3: int = 256,
        n_dec_1: int = 256,
        n_dec_2: int = 256,
        n_dec_3: int = 512,
        n_z1: int = 256,
        n_z2: int = 128,
        n_z3: int = 32,
        n_clusters: int = 10,
        n_input: int = 100,
        v: float = 1,
        device: str = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.sigma = sigma
        self.pretrain_path = pretrain_path
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z1=n_z1,
            n_z2=n_z2,
            n_z3=n_z3,
        )
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z1)
        self.gnn_5 = GNNLayer(n_z1, n_z2)
        self.gnn_6 = GNNLayer(n_z2, n_z3)
        self.gnn_7 = GNNLayer(n_z3, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        # degree
        self.v = v
        self.zinb_loss = ZINBLoss().to(device)

    def forward(self, x, adj):
        """Forward propagation.

        Parameters
        ----------
        x
            Input features.
        adj
            Adjacency matrix

        Returns
        -------
        x_bar:
            Reconstructed features.
        q
            Soft label.
        predict:
            Prediction given by softmax assignment of embedding of GCN module
        z3
            Embedding of autoencoder.
        _mean
            Data mean from ZINB.
        _disp
            Data dispersion from ZINB.
        _pi
            Data dropout probability from ZINB.
        zinb_loss:
            ZINB loss class.

        """
        # DNN Module
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        x_bar, tra1, tra2, tra3, z3, z2, z1, dec_h3 = self.ae(x)

        sigma = self.sigma
        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z1, adj)
        h = self.gnn_6((1 - sigma) * h + sigma * z2, adj)
        h = self.gnn_7((1 - sigma) * h + sigma * z3, adj, active=False)

        predict = F.softmax(h, dim=1)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss

        q = 1.0 / (1.0 + torch.sum(torch.pow(z3.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z3, _mean, _disp, _pi, zinb_loss


class GNNLayer(nn.Module):
    """GNN layer class. Construct a GNN layer with corresponding dimensions.

    Parameters
    ----------
    in_features
        Input dimension of GNN layer.
    out_features
        Output dimension of GNN layer.

    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # When passing through the network layer, the input and output variances are the same
        # including forward propagation and backward propagation

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output


class AE(nn.Module):
    """Autoencoder class.

    Parameters
    ----------
    n_enc_1
        Output dimension of encoder layer 1.
    n_enc_2
        Output dimension of encoder layer 2.
    n_enc_3
        Output dimension of encoder layer 3.
    n_dec_1
        Output dimension of decoder layer 1.
    n_dec_2
        Output dimension of decoder layer 2.
    n_dec_3
        Output dimension of decoder layer 3.
    n_input
        Input feature dimension.
    n_z1
        Output dimension of hidden layer 1.
    n_z2
        Output dimension of hidden layer 2.
    n_z3
        Output dimension of hidden layer 3.

    """

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, n_input, n_z1, n_z2, n_z3):
        super().__init__()

        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z1_layer = Linear(n_enc_3, n_z1)
        self.BN4 = nn.BatchNorm1d(n_z1)
        self.z2_layer = Linear(n_z1, n_z2)
        self.BN5 = nn.BatchNorm1d(n_z2)
        self.z3_layer = Linear(n_z2, n_z3)
        self.BN6 = nn.BatchNorm1d(n_z3)

        self.dec_1 = Linear(n_z3, n_dec_1)
        self.BN7 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN8 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN9 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        """Forward propagation.

        Parameters
        ----------
        x
            Input features.

        Returns
        -------
        x_bar
            Reconstructed features.
        enc_h1
            Output of encoder layer 1.
        enc_h2
            Output of encoder layer 2.
        enc_h3
            Output of encoder layer 3.
        z3
            Output of hidden layer 3.
        z2
            Output of hidden layer 2.
        z1
            Output of hidden layer 1.
        dec_h3
            Output of decoder layer 3.

        """
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z1 = self.BN4(self.z1_layer(enc_h3))
        z2 = self.BN5(self.z2_layer(z1))
        z3 = self.BN6(self.z3_layer(z2))

        dec_h1 = F.relu(self.BN7(self.dec_1(z3)))
        dec_h2 = F.relu(self.BN8(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN9(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z3, z2, z1, dec_h3


class RAdam(Optimizer):
    """RAdam optimizer class.

    Parameters
    ----------
    params
        Model parameters.
    lr
        Learning rate.
    betas
        Coefficients used for computing running averages of gradient and its square.
    eps
        Term added to the denominator to improve numerical stability.
    weight decay
        Weight decay (L2 penalty).
    degenerated_to_sgd
        Degenerated to SGD or not.

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2**state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /
                            (N_sma_max - 2)) / (1 - beta1**state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1**state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class MeanAct(nn.Module):
    """Mean activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation class."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
