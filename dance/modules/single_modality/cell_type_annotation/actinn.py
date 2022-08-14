"""Reimplementation of the ACTINN cell-type annotation method.

Reference
---------
Ma, Feiyang, and Matteo Pellegrini. "ACTINN: automated identification of cell types in single cell RNA sequencing."
Bioinformatics 36.2 (2020): 533-538.

"""
import itertools
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTINN(nn.Module):
    """The ACTINN cell-type classification model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        batch_size: int = 128,
        device: str = "cpu",
        hidden_dims: Tuple[int, ...] = (100, 50, 25),
        lambd: float = 0.01,
        lr: float = 0.01,
        num_epochs: int = 50,
        print_cost: bool = False,
    ):
        """Initialize the ACTINN model.

        Parameters
        ----------
        input_dim : int
            Input dimension (number of genes)
        output_dim : int
            Output dimension (number of cell types)
        hidden_dims : :obj:`tuple` of int
            Hidden layer dimensions.
        batch_size : int
            Training batch size
        device : str
            Training device
        lambd : float
            Regularization parameter
        lr : float
            Initial learning rate
        num_epochs : int
            Number of epochs to run
        print_cost : bool
            Print training loss if set to True

        """
        super().__init__()

        # Save attributes
        self.batch_size = batch_size
        self.device = device
        self.lambd = lambd
        self.lr = lr
        self.num_epochs = num_epochs
        self.print_cost = print_cost

        # Build MLP
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            *itertools.chain.from_iterable(
                zip(
                    map(nn.Linear, hidden_dims[:-1], hidden_dims[1:]),
                    itertools.repeat(nn.ReLU()),
                )),
            nn.Linear(hidden_dims[-1], output_dim),
        ).to(device)
        print(self.model)

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

        if self.print_cost:
            print(loss)

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

    def fit(self, x_train, y_train, seed=None):
        """Fit the classifier.

        Parameters
        ----------
        x_train : torch.Tensor
            training data (genes by cells).
        y_train : torch.Tensor
            training labels (cell-types by cells).
        seed : int, optional
            Random seed, if set to None, then random.

        """
        x_train = x_train.T.clone().detach().float().to(self.device)  # cells by genes
        y_train = torch.where(y_train.T)[1].to(self.device)  # cells

        # Initialize weights, optimizer, and scheduler
        self.initialize_parameters(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # Start training loop
        for epoch in range(self.num_epochs):
            epoch_seed = seed if seed is None else seed + epoch
            batches = self.random_batches(x_train, y_train, self.batch_size, epoch_seed)

            for batch_x, batch_y in batches:
                batch_cost = self.compute_loss(self.forward(batch_x), batch_y)

                optimizer.zero_grad()
                batch_cost.backward()
                optimizer.step()
                lr_scheduler.step()

        print("Parameters have been trained!")

    @torch.no_grad()
    def predict(self, x):
        """Predict cell labels.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression input features (genes by cells).

        Returns
        -------
        torch.Tensor
            Predicted cell-label indices.

        """
        x = x.T.clone().detach().to(self.device)
        z = self.forward(x)
        prediction = torch.argmax(z, dim=-1)
        return prediction

    def score(self, x, y):
        """Model performance score measured by accuracy.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression input features (genes by cells).
        y : torch.Tensor
            One-hot encoded ground truth labels (cell-types by cells).

        Returns
        -------
        float
            Prediction accuracy

        """
        pred = self.predict(x).detach().cpu()
        label = torch.where(y.T)[1]
        return (pred == label).detach().float().mean().tolist()
