"""Official release of scMoGNN method.

Reference
---------
Wen, Hongzhi, et al. "Graph Neural Networks for Multimodal Single-Cell Data Integration." arXiv preprint
arXiv:2203.01884 (2022).

"""
import math
import os
from copy import deepcopy

import dgl.nn.pytorch as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dance import logger
from dance.utils import SimpleIndexDataset
from dance.utils.metrics import batch_separated_bipartite_matching


def propagation_layer_combination(X, Y, idx, wt1, wt2, from_logits=True):
    if from_logits:
        wt1 = torch.softmax(wt1, -1)
    x = 0
    for i in range(wt1.shape[0]):
        x += wt1[i] * X[i][idx]

    if from_logits:
        wt2 = torch.softmax(wt2, -1)
    y = 0
    for i in range(wt2.shape[0]):
        y += wt2[i] * Y[i][idx]
    return x, y


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
    layers : List[List[Union[int, float]]]
        Specification of hidden layers.
    temp : int optional
        Temperature for softmax, by default to be 1.

    """

    def __init__(self, args, layers, temp=1):
        self.model = ScMoGCN(args, layers, temp).to(args.device)
        self.args = args
        wt1 = torch.tensor([0.] * (args.layers - 1)).to(args.device).requires_grad_(True)
        wt2 = torch.tensor([0.] * (args.layers - 1)).to(args.device).requires_grad_(True)
        self.wt = [wt1, wt2]

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
        if map_location is not None:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(path))

    def fit(self, g_mod1, g_mod2, labels1, labels2, train_size):
        """Fit function for training.

        Parameters
        ----------
        g_mod1 : dgl.DGLGraph
            DGLGraph for modality 1.
        g_mod2 : dgl.DGLGraph
            DGLGraph for modality 2.
        labels1 : torch.Tensor
            Column-wise matching labels.
        labels2 : torch.Tensor
            Row-wise matching labels.
        train_size : int
            Number of training samples.

        Returns
        -------
        None.

        """

        device = self.args.device
        wt = self.wt
        hcell_mod1 = cell_feature_propagation(g_mod1, layers=self.args.layers, device=self.args.device)
        hcell_mod2 = cell_feature_propagation(g_mod2, layers=self.args.layers, device=self.args.device)
        self.feat_mod1 = hcell_mod1
        self.feat_mod2 = hcell_mod2

        criterion = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()

        assert wt[0].requires_grad == wt[1].requires_grad
        opt = optim.AdamW([{
            'params': self.model.parameters()
        }, {
            'params': wt[0]
        }, {
            'params': wt[1]
        }], lr=self.args.learning_rate)

        # Make sure the batch size is small enough to cover all splits
        BATCH_SIZE = min(4096, math.floor(train_size / 2))

        idx = torch.randperm(train_size)
        train_idx = idx[:-BATCH_SIZE]
        val_idx = idx[-BATCH_SIZE:]
        test_idx = np.arange(train_size, hcell_mod1[0].shape[0])
        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        maxval = -1
        vals = []
        for epoch in range(self.args.epochs):
            self.model.train()
            logger.info(f'epoch {epoch}')
            total_loss = 0
            accum_acc = [0, 0]

            for step, batch_idx in enumerate(train_loader):
                X, Y = propagation_layer_combination(hcell_mod1, hcell_mod2, batch_idx, wt[0], wt[1])

                logits = self.model(X, Y)
                temp = torch.arange(logits.shape[0]).to(logits.device)
                loss = criterion(logits, temp) + criterion(logits.T, temp)

                forward_accuracy = (torch.argmax(logits, dim=1) == temp).float().mean().item()
                backward_accuracy = (torch.argmax(logits, dim=0) == temp).float().mean().item()
                accum_acc[0] += forward_accuracy
                accum_acc[1] += backward_accuracy

                emb1, emb2 = self.model.encode(X, Y)
                pred1, pred2 = self.model.decode(emb2, emb1)
                rec1, rec2 = self.model.decode(emb1, emb2)

                loss2 = criterion2(pred1, X) + criterion2(pred2, Y)
                loss3 = criterion2(rec1, X) + criterion2(rec2, Y)

                total_loss += loss.item()

                if self.args.auxiliary_loss > 0:
                    loss = loss + loss2 + loss3

                opt.zero_grad()
                loss.backward()
                opt.step()

            logger.info('training loss: %.5f, forward: %.4f, backward: %.4f', total_loss / len(train_loader),
                        accum_acc[0] / len(train_loader), accum_acc[1] / len(train_loader))

            temp = torch.arange(val_idx.shape[0]).to(device)
            vals.append(self.score(val_idx, labels1=temp, labels2=temp))
            logger.info('validation score: %.5f', vals[-1])
            if epoch % 10 == 9:
                logger.info('testing score: %.5f', self.score(test_idx, labels1=labels1, labels2=labels2))

            if vals[-1] > maxval:
                maxval = vals[-1]
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(self.model.state_dict(), f'models/model_{self.args.seed}.pth')
                best_dict = deepcopy(self.model.state_dict())
                weight_record = [wt[0].detach(), wt[1].detach()]

            if max(vals) != max(vals[-20:]):
                logger.info('Early stopped.')
                break

        logger.info(f'Valid: {maxval}')

        self.wt = weight_record
        self.model.load_state_dict(best_dict)
        return self

    def predict(self, idx, enhance=False, batch1=None, batch2=None, threshold_quantile=0.95):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        idx : Iterable[int]
            Cell indices for prediction.
        enhance : bool optional
            Whether enable enhancement matching (e.g. bipartite matching), by default to be False.
        batch1 : torch.Tensor optional
            Batch labels of modality 1, by default to be None.
        batch2 : torch.Tensor optional
            Batch labels of modality 2, by default to be None.
        threshold_quantile: float
            Parameter for batch_separated_bipartite_matching when enhance is set to true, which controls the sparsity.

        Returns
        -------
        pred : torch.Tensor
            Predicted matching matrix.

        """
        # inputs: [train_mod1, train_mod2], idx: valid_idx, labels: [sol, sol.T], wt: [wt0, wt1]
        self.model.eval()

        with torch.no_grad():
            wt = self.wt
            m1, m2 = propagation_layer_combination(self.feat_mod1, self.feat_mod2, idx, wt[0], wt[1])

            if not enhance:
                pred = self.model(m1, m2)
                return pred

            else:
                emb1, emb2 = self.model.encode(m1, m2)
                pred = batch_separated_bipartite_matching(batch1[idx], batch2[idx], emb1, emb2, threshold_quantile)
                return pred

    def score(self, idx, labels1=None, labels2=None, labels_matrix=None, enhance=False, batch1=None, batch2=None,
              threshold_quantile=0.95):
        """Score function to get score of prediction.

        Parameters
        ----------
        idx : Iterable[int]
            Index of testing cells for scoring.
        labels1 : torch.Tensor
            Column-wise matching labels.
        labels2 : torch.Tensor
            Row-wise matching labels.
        labels_matrix: torch.Tensor
            Matching labels.
        enhance : bool optional
            Whether enable enhancement matching (e.g. bipartite matching), by default to be False.
        batch1 : torch.Tensor optional
            Batch labels of modality 1, by default to be None.
        batch2 : torch.Tensor optional
            Batch labels of modality 2, by default to be None.
        threshold_quantile: float
            Parameter for batch_separated_bipartite_matching when enhance is set to true, which controls the sparsity.

        Returns
        -------
        score : float
            Accuracy of predicted matching between two modalities.

        """

        if not enhance:

            logits = self.predict(idx, enhance, batch1, batch2)
            backward_accuracy = (torch.argmax(logits, dim=0) == labels1).float().mean().item()
            forward_accuracy = (torch.argmax(logits, dim=1) == labels2).float().mean().item()
            return (forward_accuracy + backward_accuracy) / 2

        else:

            matrix = self.predict(idx, enhance, batch1, batch2, threshold_quantile)
            score = (matrix * labels_matrix.numpy()).sum() / labels_matrix.shape[0]

            return score


class ScMoGCN(nn.Module):

    def __init__(self, args, layers, temp=1):
        super().__init__()

        assert (len(layers) == 4)
        self.nn = [list() for i in range(4)]
        self.temp = temp
        self.args = args

        for j, shape in enumerate(layers):
            for i, s in enumerate(shape):
                self.nn[j].append(nn.Linear(s[0], s[1]))
                if i < len(shape) - 1:
                    self.nn[j].append(nn.GELU())
                    if len(s) == 3:
                        self.nn[j].append(nn.Dropout(s[2]))

        self.nn = [nn.Sequential(*n) for n in self.nn]
        for i, n in enumerate(self.nn):
            self.add_module(str(i), n)

    def encode(self, m1, m2):
        emb1 = self.nn[0](m1)
        emb2 = self.nn[1](m2)
        emb1 = emb1 / torch.norm(emb1, p=2, dim=-1, keepdim=True)
        emb2 = emb2 / torch.norm(emb2, p=2, dim=-1, keepdim=True)
        return emb1, emb2

    def decode(self, emb1, emb2):
        return self.nn[2](emb1), self.nn[3](emb2)

    def forward(self, m1, m2):
        emb1, emb2 = self.encode(m1, m2)
        return torch.matmul(emb1, emb2.T) * math.exp(self.temp)
