"""Official release of scMoGNN method.

Reference
---------
Wen, Hongzhi, et al. "Graph Neural Networks for Multimodal Single-Cell Data Integration." arXiv preprint arXiv:2203.01884 (2022).

"""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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


class ScMoGCNWrapper:
    """ScMoGCN class.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace object that contains arguments of ScMoGCN. For details of parameters in parser args, please refer to link (parser help document).
    layers : list[int]
        Specification of dimensions of hidden layers.
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

    def fit(self, dataset, feats, labels):
        """fit function for training.

        Parameters
        ----------
        dataset : dance.datasets.multimodality.ModalityMatchingNIPSDataset
            Dataset for mdality matching.
        feats : torch.Tensor
            Modality features.

        Returns
        -------
        None.

        """

        device = self.args.device
        wt = self.wt
        hcell_mod1, hcell_mod2 = feats

        labels0, labels1 = labels

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

        BATCH_SIZE = 4096

        idx = torch.randperm(dataset.sparse_features()[0].shape[0])
        train_idx = idx[:-BATCH_SIZE]
        val_idx = idx[-BATCH_SIZE:]
        test_idx = np.arange(dataset.sparse_features()[0].shape[0],
                             dataset.sparse_features()[0].shape[0] + dataset.sparse_features()[2].shape[0])
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
            print('epoch', epoch)
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

            print('training loss: %.5f, foward: %.4f, backward: %.4f' %
                  (total_loss / len(train_loader), accum_acc[0] / len(train_loader), accum_acc[1] / len(train_loader)))

            temp = torch.arange(val_idx.shape[0]).to(device)
            vals.append(self.score([hcell_mod1, hcell_mod2], val_idx, [temp, temp]))
            print('validation score: %.5f' % vals[-1])
            if epoch % 10 == 9:
                print('testing score: %.5f' % self.score([hcell_mod1, hcell_mod2], test_idx, [labels0, labels1]))

            if vals[-1] > maxval:
                maxval = vals[-1]
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(self.model.state_dict(), f'models/model_{self.args.rnd_seed}.pth')
                weight_record = [wt[0].detach(), wt[1].detach()]

            if max(vals) != max(vals[-20:]):
                print('Early stopped.')
                break

        print('Valid: ', maxval)

        self.wt = weight_record
        return self

    def predict(self, inputs, idx, enhance=False, dataset=None):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Multimodality features.
        idx : Iterable[int]
            Cell indices for prediction.
        enhance : bool optional
            Whether enable enhancement matching (e.g. bipartite matching), by default to be False.
        dataset : dance.datasets.multimodality.ModalityMatchingNIPSDataset optional
            Dataset for modality matching, needed when enhance parameter set to be True.

        Returns
        -------
        pred : torch.Tensor
            Predicted matching matrix.

        """
        # inputs: [train_mod1, train_mod2], idx: valid_idx, labels: [sol, sol.T], wt: [wt0, wt1]
        self.model.eval()

        with torch.no_grad():
            wt = self.wt
            m1, m2 = propagation_layer_combination(inputs[0], inputs[1], idx, wt[0], wt[1])

            if not enhance:
                pred = self.model(m1, m2)
                return pred

            else:
                emb1, emb2 = self.model.encode(m1, m2)
                pred = batch_separated_bipartite_matching(dataset, emb1, emb2)
                return pred

    def score(self, inputs, idx, labels, enhance=False, dataset=None):
        """Score function to get score of prediction.

        Parameters
        ----------
        inputs : list[torch.Tensor]
            Multimodality features.
        idx : Iterable[int]
            Index of testing cells for scoring.
        labels : torch.Tensor
            Ground truth label of cell matching matrix
        enhance : bool optional
            Whether enable enhancement matching (e.g. bipartite matching), by default to be False.
        dataset : dance.datasets.multimodality.ModalityMatchingNIPSDataset optional
            Dataset for modality matching, needed when enhance parameter set to be True.

        Returns
        -------
        score : float
            Accuracy of predicted matching between two modalities.

        """

        if not enhance:

            logits = self.predict(inputs, idx, enhance, dataset)
            forward_accuracy = (torch.argmax(logits, dim=1) == labels[1]).float().mean().item()
            backward_accuracy = (torch.argmax(logits, dim=0) == labels[0]).float().mean().item()

            return (forward_accuracy + backward_accuracy) / 2

        else:

            matrix = self.predict(inputs, idx, enhance, dataset)
            score = (matrix * labels.numpy()).sum() / labels.shape[0]

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
