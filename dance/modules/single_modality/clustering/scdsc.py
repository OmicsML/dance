"""Reimplementation of scDSC.

Extended from https://github.com/DHUDBlab/scDSC

Reference
----------
Gan, Yanglan, et al. "Deep structural clustering for single-cell RNA-seq data jointly through autoencoder and graph
neural network." Briefings in Bioinformatics 23.2 (2022): bbac018.

"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dance.transforms.preprocess import load_graph
from dance.utils.loss import ZINBLoss
from dance.utils.metrics import cluster_acc


class SCDSCWrapper:
    """scDSC wrapper class.

    Parameters
    ----------
    args : argparse.Namespace
        a Namespace contains arguments of scDSC. For details of parameters in parser args, please refer to link (parser help document).

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.model = SCDSC(args).to(self.device)
        self.model_pre = AE(n_enc_1=args.n_enc_1, n_enc_2=args.n_enc_2, n_enc_3=args.n_enc_3, n_dec_1=args.n_dec_1,
                            n_dec_2=args.n_dec_2, n_dec_3=args.n_dec_3, n_input=args.n_input, n_z1=args.n_z1,
                            n_z2=args.n_z2, n_z3=args.n_z3).to(self.device)

    def target_distribution(self, q):
        """Calculate auxiliary target distribution p with q.

        Parameters
        ----------
        q :
            soft label.

        Returns
        -------
        p :
            target distribution.

        """
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def pretrain_ae(self, dataset, batch_size, n_epochs, fname, lr=1e-3):
        """Pretrain autoencoder.

        Parameters
        ----------
        dataset :
            input dataset.
        batch_size : int
            size of batch.
        n_epochs : int
            number of epochs.
        lr : float optional
            learning rate.
        fname : str
            path to save autoencoder weights.

        Returns
        -------
        None.

        """
        device = self.device
        train_loader = DataLoader(dataset, batch_size, shuffle=True)
        model = self.model_pre
        optimizer = Adam(model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for batch_idx, (x, _) in enumerate(train_loader):
                x = x.to(device)
                x_bar, _, _, _, _, _, _, _ = model(x)

                x_bar = x_bar.cpu()
                x = x.cpu()
                loss = F.mse_loss(x_bar, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                x = torch.Tensor(dataset.x).to(device).float()
                x_bar, _, _, _, z3, _, _, _ = model(x)
                loss = F.mse_loss(x_bar, x)
                print('Pretrain epoch %3d, MSE loss: %.8f' % (epoch + 1, loss))

        torch.save(model.state_dict(), fname)

    def fit(self, dataset, X_raw, sf, graph_path, lr=1e-03, n_epochs=300, bcl=0.1, cl=0.01, rl=1, zl=0.1):
        """Train model.

        Parameters
        ----------
        dataset :
            input dataset.
        X_raw :
            raw input features.
        sf : list
            scale factor of ZINB loss.
        graph_path : str
            path of graph file.
        lr : float optional
            learning rate.
        n_epochs : int optional
            number of epochs.
        bcl : float optional
            parameter of binary crossentropy loss.
        cl : float optional
            parameter of Kullback–Leibler divergence loss.
        rl : float optional
            parameter of reconstruction loss.
        zl : float optional
            parameter of ZINB loss.

        Returns
        -------
        None.

        """
        device = self.device
        model = self.model
        optimizer = Adam(model.parameters(), lr=lr)
        # optimizer = RAdam(model.parameters(), lr=lr)

        adj = load_graph(graph_path, dataset.x)
        adj = adj.to(device)
        X_raw = torch.tensor(X_raw).to(device)
        sf = torch.tensor(sf).to(device)
        data = torch.Tensor(dataset.x).to(device)
        y = dataset.y

        aris = []
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
                    print("Epoch %3d, ARI: %.4f, Best ARI: %.4f" % (epoch + 1, ari, max(aris)))

                    p_ = {f'epoch{epoch}': p}
                    q_ = {f'epoch{epoch}': tmp_q}
                    P = {**P, **p_}
                    Q = {**Q, **q_}

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
        self.q = Q[f'epoch{index}']

    def predict(self):
        """Get predictions from the trained model.

        Parameters
        ----------
        None.

        Returns
        -------
        y_pred : np.array
            prediction of given clustering method.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        return y_pred

    def score(self, y):
        """Evaluate the trained model.

        Parameters
        ----------
        y : list
            true labels.

        Returns
        -------
        acc : float
            accuracy.
        nmi : float
            normalized mutual information.
        ari : float
            adjusted Rand index.

        """
        y_pred = torch.argmax(self.q, dim=1).data.cpu().numpy()
        acc = np.round(cluster_acc(y, y_pred), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
        return acc, nmi, ari


class SCDSC(nn.Module):
    """scDSC class.

    Parameters
    ----------
    args : argparse.Namespace
        a Namespace contains arguments of GCNAE. For details of parameters in parser args, please refer to link (parser help document).
    device : str
        computing device.
    sigma : float
        balance parameter.
    pretrain_path : str
        path of saved autoencoder weights.
    n_enc_1 : int
        output dimension of encoder layer 1.
    n_enc_2 : int
        output dimension of encoder layer 2.
    n_enc_3 : int
        output dimension of encoder layer 3.
    n_dec_1 : int
        output dimension of decoder layer 1.
    n_dec_2 : int
        output dimension of decoder layer 2.
    n_dec_3 : int
        output dimension of decoder layer 3.
    n_z1 : int
        output dimension of hidden layer 1.
    n_z2 : int
        output dimension of hidden layer 2.
    n_z3 : int
        output dimension of hidden layer 3.
    n_clusters : int
        number of clusters.
    n_input : int
        input feature dimension.
    v : float
        parameter of soft assignment.

    """

    def __init__(self, args):
        super().__init__()
        device = args.device
        self.sigma = args.sigma
        self.pretrain_path = args.pretrain_path
        self.ae = AE(
            n_enc_1=args.n_enc_1,
            n_enc_2=args.n_enc_2,
            n_enc_3=args.n_enc_3,
            n_dec_1=args.n_dec_1,
            n_dec_2=args.n_dec_2,
            n_dec_3=args.n_dec_3,
            n_input=args.n_input,
            n_z1=args.n_z1,
            n_z2=args.n_z2,
            n_z3=args.n_z3,
        )
        self.gnn_1 = GNNLayer(args.n_input, args.n_enc_1)
        self.gnn_2 = GNNLayer(args.n_enc_1, args.n_enc_2)
        self.gnn_3 = GNNLayer(args.n_enc_2, args.n_enc_3)
        self.gnn_4 = GNNLayer(args.n_enc_3, args.n_z1)
        self.gnn_5 = GNNLayer(args.n_z1, args.n_z2)
        self.gnn_6 = GNNLayer(args.n_z2, args.n_z3)
        self.gnn_7 = GNNLayer(args.n_z3, args.n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(args.n_clusters, args.n_z3))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(args.n_dec_3, args.n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(args.n_dec_3, args.n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(args.n_dec_3, args.n_input), nn.Sigmoid())
        # degree
        self.v = args.v
        self.zinb_loss = ZINBLoss().to(device)

    def forward(self, x, adj):
        """Forward propagation.

        Parameters
        ----------
        x :
            input features.
        adj :
            adjacency matrix

        Returns
        -------
        x_bar:
            reconstructed features.
        q :
            soft label.
        predict:
            prediction given by softmax assignment of embedding of GCN module
        z3 :
            embedding of autoencoder.
        _mean :
            data mean from ZINB.
        _disp :
            data dispersion from ZINB.
        _pi :
            data dropout probability from ZINB.
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
    in_features : int
        input dimension of GNN layer.
    out_features : int
        output dimension of GNN layer.

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
    n_enc_1 : int
        output dimension of encoder layer 1.
    n_enc_2 : int
        output dimension of encoder layer 2.
    n_enc_3 : int
        output dimension of encoder layer 3.
    n_dec_1 : int
        output dimension of decoder layer 1.
    n_dec_2 : int
        output dimension of decoder layer 2.
    n_dec_3 : int
        output dimension of decoder layer 3.
    n_input : int
        input feature dimension.
    n_z1 : int
        output dimension of hidden layer 1.
    n_z2 : int
        output dimension of hidden layer 2.
    n_z3 : int
        output dimension of hidden layer 3.

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
        x :
            input features.

        Returns
        -------
        x_bar:
            reconstructed features.
        enc_h1:
            output of encoder layer 1.
        enc_h2:
            output of encoder layer 2.
        enc_h3:
            output of encoder layer 3.
        z3 :
            output of hidden layer 3.
        z2 :
            output of hidden layer 2.
        z1 :
            output of hidden layer 1.
        dec_h3 :
            output of decoder layer 3.

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
    params :
        model parameters.
    lr : float optional
        learning rate.
    betas : tuple optional
        coefficients used for computing running averages of gradient and its square.
    eps : float optional
        term added to the denominator to improve numerical stability.
    weight decay : float optional
        weight decay (L2 penalty).
    degenerated_to_sgd : bool optional
        degenerated to SGD or not.

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
