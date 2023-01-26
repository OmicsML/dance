"""Reimplementation of the ACTINN cell-type annotation method.

Reference
---------
Ma, Feiyang, and Matteo Pellegrini. "ACTINN: automated identification of cell types in single cell RNA sequencing."
Bioinformatics 36.2 (2020): 533-538.

"""
import itertools
from typing import Tuple

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F

from dance.transforms import AnnDataTransform, Compose, FilterGenesPercentile, SetConfig
from dance.typing import LogLevel, Optional


class ACTINN(nn.Module):
    """The ACTINN cell-type classification model.

    Parameters
    ----------
    hidden_dims : :obj:`tuple` of int
        Hidden layer dimensions.
    lambd : float
        Regularization parameter
    device : str
        Training device

    """

    def __init__(
            self,
            *,
            hidden_dims: Tuple[int, ...] = (100, 50, 25),
            lambd: float = 0.01,
            device: str = "cpu",
    ):
        super().__init__()

        # Save attributes
        self.hidden_dims = hidden_dims
        self.device = device
        self.lambd = lambd

    def _init_model(self, input_dim: int, output_dim: int):
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dims[0]),
            nn.ReLU(),
            *itertools.chain.from_iterable(
                zip(
                    map(nn.Linear, self.hidden_dims[:-1], self.hidden_dims[1:]),
                    itertools.repeat(nn.ReLU()),
                )),
            nn.Linear(self.hidden_dims[-1], output_dim),
        ).to(self.device)
        print(self.model)

    def preprocess(self, data, /, **kwargs):
        self.preprocessing_pipeline(**kwargs)(data)

    @staticmethod
    def preprocessing_pipeline(normalize: bool = True, filter_genes: bool = True, log_level: LogLevel = "INFO"):
        transforms = []

        if normalize:
            transforms.append(AnnDataTransform(sc.pp.normalize_total, target_sum=1e4))
            transforms.append(AnnDataTransform(sc.pp.log1p, base=2))

        if filter_genes:
            transforms.append(AnnDataTransform(sc.pp.filter_genes, min_cells=1))
            transforms.append(FilterGenesPercentile(min_val=1, max_val=99, mode="sum"))
            transforms.append(FilterGenesPercentile(min_val=1, max_val=99, mode="cv"))

        transforms.append(SetConfig({"label_channel": "cell_type"}))

        return Compose(*transforms, log_level=log_level)

    def forward(self, x):
        """Forward propagation."""
        return self.model(x)

    @torch.no_grad()
    def initialize_parameters(self, seed=None):
        """Initialize parameters."""
        if seed is not None:
            torch.manual_seed(seed)

        for i in range(0, len(self.model), 2):
            nn.init.xavier_normal_(self.model[i].weight)
            self.model[i].bias[:] = 0

    def compute_loss(self, z, y):
        """Compute loss function.

        Parameters
        ----------
        z : torch.Tensor
            Output of forward propagation (cells by cell-types).
        y : torch.Tensor
            Cell labels (cells).

        Returns
        -------
        torch.Tensor
            Loss.

        """
        log_prob = F.log_softmax(z, dim=-1)
        loss = nn.NLLLoss()(log_prob, y)
        for i in range(0, len(self.model), 2):  # TODO: replace with weight_decay
            loss += self.lambd * torch.sum(self.model[i].weight**2) / 2

        return loss

    def random_batches(self, x, y, batch_size=32, seed=None):
        """Shuffle data and split into batches.

        Parameters
        ----------
        x : torch.Tensor
            Training data (cells by genes).
        y : torch.Tensor
            True labels (cells by cell-types).

        Yields
        ------
        Tuple[torch.Tensor, torch.Tensor]
            Batch of training data (x, y).

        """
        ns = x.shape[0]
        perm = np.random.default_rng(seed).permutation(ns).tolist()
        slices = [perm[i:i + batch_size] for i in range(0, ns, batch_size)]
        yield from map(lambda slice_: (x[slice_], y[slice_]), slices)

    def fit(
        self,
        x_train,
        y_train,
        *,
        batch_size: int = 128,
        lr: float = 0.01,
        num_epochs: int = 50,
        print_cost: bool = False,
        seed: Optional[int] = None,
    ):
        """Fit the classifier.

        Parameters
        ----------
        x_train : torch.Tensor
            training data (cells by genes).
        y_train : torch.Tensor
            training labels (cells by cell-types).
        batch_size : int
            Training batch size
        lr : float
            Initial learning rate
        num_epochs : int
            Number of epochs to run
        print_cost : bool
            Print training loss if set to True
        seed : int, optional
            Random seed, if set to None, then random.

        """
        input_dim, output_dim = x_train.shape[1], y_train.shape[1]
        x_train = x_train.clone().detach().float().to(self.device)  # cells by genes
        y_train = torch.where(y_train)[1].to(self.device)  # cells

        # Initialize weights, optimizer, and scheduler
        self._init_model(input_dim, output_dim)
        self.initialize_parameters(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # Start training loop
        for epoch in range(num_epochs):
            epoch_seed = seed if seed is None else seed + epoch
            batches = self.random_batches(x_train, y_train, batch_size, epoch_seed)

            tot_cost = tot_size = 0
            for batch_x, batch_y in batches:
                batch_cost = self.compute_loss(self.forward(batch_x), batch_y)
                tot_cost += batch_cost.item()
                tot_size += 1

                optimizer.zero_grad()
                batch_cost.backward()
                optimizer.step()
                lr_scheduler.step()

            if (epoch % 10 == 0) and print_cost:
                print(f"Epoch: {epoch:>4d} Loss: {tot_cost / tot_size:6.4f}")

        print("Parameters have been trained!")

    @torch.no_grad()
    def predict(self, x):
        """Predict cell labels.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression input features (cells by genes).

        Returns
        -------
        torch.Tensor
            Predicted cell-label indices.

        """
        x = x.clone().detach().to(self.device)
        z = self.forward(x)
        prediction = torch.argmax(z, dim=-1)
        return prediction

    def score(self, pred, true):
        """Model performance score measured by accuracy.

        Parameters
        ----------
        pred : torch.Tensor
            Gene expression input features (cells by genes).
        true : torch.Tensor
            Encoded ground truth cell type labels (cells by cell-types).

        Returns
        -------
        float
            Prediction accuracy.

        """
        return true[range(pred.shape[0]), pred.squeeze(-1)].detach().mean().item()
