"""Reimplementation of the ACTINN cell-type annotation method.

Reference
---------
Ma, Feiyang, and Matteo Pellegrini. "ACTINN: automated identification of cell types in single cell RNA sequencing."
Bioinformatics 36.2 (2020): 533-538.

"""
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from dance.models.nn import VanillaMLP
from dance.modules.base import BaseClassificationMethod
from dance.transforms import AnnDataTransform, Compose, FilterGenesPercentile, SetConfig
from dance.typing import LogLevel, Optional, Tuple


class ACTINN(BaseClassificationMethod):
    """The ACTINN cell-type classification model.

    Parameters
    ----------
    hidden_dims
        Hidden layer dimensions.
    lambd
        Regularization parameter
    device
        Training device

    """

    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...] = (100, 50, 25),
        lambd: float = 0.01,
        device: str = "cpu",
        random_seed: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.lambd = lambd
        self.device = device
        self.random_seed = random_seed

        self.model_size = len(hidden_dims) + 2

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

    def compute_loss(self, z: Tensor, y: Tensor):
        """Compute loss function.

        Parameters
        ----------
        z
            Output of forward propagation (cells by cell-types).
        y
            Cell labels (cells).

        Returns
        -------
        torch.Tensor
            Loss.

        """
        log_prob = F.log_softmax(z, dim=-1)
        loss = nn.NLLLoss()(log_prob, y)
        for i, p in enumerate(self.model.model):
            if (i % 2) == 0:  # skip activation layers
                loss += self.lambd * (p.weight**2).sum() / 2
        return loss

    def random_batches(self, x: Tensor, y: Tensor, batch_size: int = 32, seed: Optional[int] = None):
        """Shuffle data and split into batches.

        Parameters
        ----------
        x
            Training data (cells by genes).
        y
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
        x_train: Tensor,
        y_train: Tensor,
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
        x_train
            training data (cells by genes).
        y_train
            training labels (cells by cell-types).
        batch_size
            Training batch size.
        lr
            Initial learning rate.
        num_epochs
            Number of epochs to run.
        print_cost
            Print training loss if set to True.
        seed
            Random seed, if set to None, then random.

        """
        input_dim, output_dim = x_train.shape[1], y_train.shape[1]
        x_train = x_train.clone().detach().float().to(self.device)  # cells by genes
        y_train = torch.where(y_train)[1].to(self.device)  # cells

        # Initialize weights, optimizer, and scheduler
        self.model = VanillaMLP(input_dim, output_dim, hidden_dims=self.hidden_dims, device=self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

        # Start training loop
        global_steps = 0
        for epoch in range(num_epochs):
            epoch_seed = seed if seed is None else seed + epoch
            batches = self.random_batches(x_train, y_train, batch_size, epoch_seed)

            tot_cost = tot_size = 0
            for batch_x, batch_y in batches:
                batch_cost = self.compute_loss(self.model(batch_x), batch_y)
                tot_cost += batch_cost.item()
                tot_size += 1

                optimizer.zero_grad()
                batch_cost.backward()
                optimizer.step()

                global_steps += 1
                if global_steps % 1000 == 0:
                    lr_scheduler.step()

            if print_cost and (epoch % 10 == 0):
                print(f"Epoch: {epoch:>4d} Loss: {tot_cost / tot_size:6.4f}")

    @torch.no_grad()
    def predict(self, x: Tensor):
        """Predict cell labels.

        Parameters
        ----------
        x
            Gene expression input features (cells by genes).

        Returns
        -------
        torch.Tensor
            Predicted cell-label indices.

        """
        x = x.clone().detach().to(self.device)
        z = self.model(x)
        prediction = torch.argmax(z, dim=-1)
        return prediction
