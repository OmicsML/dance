"""Reimplementation of DeepImpute.

Extended from https://github.com/lanagarmire/DeepImpute

Reference
----------
Arisdakessian, Cédric, et al. "DeepImpute: an accurate, fast, and scalable deep neural network method to impute
single-cell RNA-seq data." Genome biology 20.1 (2019): 1-14.

"""

import tempfile
from math import floor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset

from dance.modules.base import BaseRegressionMethod
from dance.transforms import (AnnDataTransform, CellwiseMaskData, Compose, FilterCellsScanpy, FilterGenesScanpy,
                              GeneHoldout, SaveRaw, SetConfig)
from dance.typing import Any, List, LogLevel, Optional, Tuple


class NeuralNetworkModel(nn.Module):
    """Model class.

    Parameters
    ----------
    None
    Returns
    -------
    None

    """

    def __init__(self, inputdim, sub_outputdim, hidden_dim=None, dropout=0.2):
        super().__init__()
        if (hidden_dim is None):
            hidden_dim = floor(sub_outputdim / 2)
        self.layer1 = nn.Linear(inputdim, hidden_dim)
        self.layer2 = nn.Dropout(p=dropout)
        self.layer3 = nn.Linear(hidden_dim, sub_outputdim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        x = F.softplus(self.layer3(x))
        return (x)


class DeepImpute(nn.Module, BaseRegressionMethod):
    """DeepImpute class.

    Parameters
    ----------
    learning_rate : float optional
        learning rate
    batch_size : int optional
        batch size
    max_epochs : int optional
        maximum epochs

    patience : int optional
        number of epochs before stopping once loss stops to improve
    gpu : int optional
        option to use gpu
    loss : string optional
        loss function
    output_prefix : string optinal
        directory to save outputs
    sub_outputdim : int optional
        output dimensions in each subnetwork
    hidden_dim : int optional
        dimension of the dense layer in each subnetwork

    verbose: int optional
        verbose option

    seed: int optional
        random seed
    architecture: optional
        network architecture

    imputed_only: boolean optional
        whether to return imputed genes only

    policy: string optional
        imputation policy

    """

    def __init__(self, predictors, targets, dataset, sub_outputdim=512, hidden_dim=256, dropout=0.2, seed=1, gpu=-1):
        super().__init__()
        self.seed = seed
        self.predictors = predictors
        self.targets = targets
        self.dataset = dataset
        self.sub_outputdim = sub_outputdim
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.prj_path = Path().resolve()
        self.save_path = self.prj_path / "deepimpute"
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device(f'cuda:{gpu}' if gpu != -1 and torch.cuda.is_available() else 'cpu')
        self.models = self.build([len(genes) for genes in predictors], [len(genes) for genes in targets], self.device)

    @staticmethod
    def preprocessing_pipeline(min_cells: float = 0.1, n_top: int = 5, sub_outputdim: int = 512, mask: bool = True,
                               distr: str = "exp", mask_rate: float = 0.1, seed: int = 1, log_level: LogLevel = "INFO"):

        transforms = [
            FilterGenesScanpy(min_cells=min_cells),
            FilterCellsScanpy(min_counts=1),
            SaveRaw(),
            AnnDataTransform(sc.pp.log1p),
            GeneHoldout(n_top=n_top, batch_size=sub_outputdim),
        ]
        if mask:
            transforms.extend([
                CellwiseMaskData(distr=distr, mask_rate=mask_rate, seed=seed),
                SetConfig({
                    "feature_channel": [None, None, "targets", "predictors", "train_mask"],
                    "feature_channel_type": ["X", "raw_X", "uns", "uns", "layers"],
                    "label_channel": [None, None],
                    "label_channel_type": ["X", "raw_X"],
                })
            ])
        else:
            transforms.extend([
                SetConfig({
                    "feature_channel": [None, None, "targets", "predictors"],
                    "feature_channel_type": ["X", "raw_X", "uns", "uns"],
                    "label_channel": [None, None],
                    "label_channel_type": ["X", "raw_X"],
                })
            ])

        return Compose(*transforms, log_level=log_level)

    def wMSE(self, y_true, y_pred, binary=False):
        """Weighted MSE.

        Parameters
        ----------
        y_true: array
            true expression
        Y_train: array
            predicted expression
        binary: boolean optional
            whether to use binary weights
        Returns
        -------
        val: float
            weighted MSE

        """

        if binary:
            tmp = y_true > 0
            weights = tmp.type(torch.FloatTensor)
        else:
            weights = y_true
        val = torch.mean(weights * torch.square(y_true - y_pred))
        return val

    def build(self, inputdims, outputdims, device="cpu"):
        """Build model.

        Parameters
        ----------
        inputdims: int
            number of neurons as input in the first layer
        Returns
        -------
        models : array
            array of subnetworks

        """
        models = []
        for i in range(len(inputdims)):
            models.append(
                NeuralNetworkModel(inputdims[i], outputdims[i], hidden_dim=self.hidden_dim,
                                   dropout=self.dropout).to(device))

        return models

    def maskdata(self, X, mask, idx=None):
        if idx is None:
            idx = range(len(X))
        submask = mask[idx]
        X_masked = torch.zeros_like(X).to(X.device)
        X_masked[submask] = X[submask]
        counter_submask = ~submask

        return X_masked, submask, counter_submask

    def fit(self, X, Y, mask=None, batch_size=64, lr=1e-3, n_epochs=100, patience=5, train_idx=None):
        """Train model.

        Parameters
        ----------
        X_train: optional
            Training data including input genes
        Y_train: optional
            Training data including target genes to be inputed
        X_valid:  optional
            Validation data including input predictor genes
        Y_valid:  optional
            Validation data including target genes to be inputed
        predictors: array optional
            input genes as predictors for target genes
        Returns
        -------
        None

        """
        predictors = self.predictors
        targets = self.targets
        device = self.device

        # Specify train validation split
        if mask is not None:
            X_train, _, valid_mask = self.maskdata(X, mask, train_idx)
            X_valid = X_train
            Y_valid = Y_train = Y
        else:
            rng = np.random.default_rng(self.seed)
            train_idx_permuted = rng.permutation(range(len(X)))
            train_idx = train_idx_permuted[:int(len(train_idx_permuted) * 0.9)]
            valid_idx = train_idx_permuted[int(len(train_idx_permuted) * 0.9):]
            X_train = X[train_idx]
            X_valid = X[valid_idx]
            Y_train = Y[train_idx]
            Y_valid = Y[valid_idx]
            valid_mask = np.ones_like(X_valid.cpu()).astype(bool)

        X_train_list, X_valid_list, Y_train_list, Y_valid_list, valid_mask_list = [], [], [], [], []
        for j, inputgenes in enumerate(predictors):
            X_train_list.append(X_train[:, inputgenes])
            X_valid_list.append(X_valid[:, inputgenes])
            Y_train_list.append(Y_train[:, targets[j]])
            Y_valid_list.append(Y_valid[:, targets[j]])
            valid_mask_list.append(valid_mask[:, targets[j]])

        data = [TensorDataset(X_train_list[i], Y_train_list[i]) for i in range(len(predictors))]
        train_loaders = [DataLoader(data[i], batch_size=batch_size, shuffle=True) for i in range(len(data))]
        optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]

        for i, model in enumerate(self.models):
            optimizer = optimizers[i]
            train_loader = train_loaders[i]
            val_losses = []
            counter = 0
            for epoch in range(n_epochs):
                model.train()
                train_loss = 0
                for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                    y_pred = model(x_batch.to(device))
                    loss = self.wMSE(y_batch.to(device), y_pred)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(x_batch)
                train_loss = train_loss / len(X_train_list[i])

                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.Tensor(X_valid_list[i]).to(device))
                    val_loss = F.mse_loss(y_pred[valid_mask_list[i]],
                                          Y_valid_list[i].to(device)[valid_mask_list[i]]).item()
                print("Model {:d}, epoch {:d}, train loss: {:f}, valid loss: {:f}.".format(
                    i, epoch, train_loss, val_loss))

                val_losses.append(val_loss)
                min_val = min(val_losses)
                if val_loss == min_val:
                    self.save_model(model, optimizer, i)
                else:
                    counter += 1
                    if counter == patience:
                        print("Early stopped")
                        break

    def save_model(self, model, optimizer, i):
        """Save model.

        Parameters
        ----------
        model:
            model to be saved

        optimizer:
            optimizer

        i: int
            index of the subnetwork to be loaded
        Returns
        -------
        None

        """

        model_string = 'model_' + str(i)
        opt_string = 'optimizer_' + str(i)
        state = {model_string: model.state_dict(), opt_string: optimizer.state_dict()}
        torch.save(state, self.save_path / f"{self.dataset}_{i}.pt")

    def load_model(self, model, i):
        """Load model.

        Parameters
        ----------
        model:
            model to be loaded

        i: int
            index of the subnetwork to be loaded
        Returns
        -------
        model :
            loaded model

        """

        model_path = self.save_path / f"{self.dataset}_{i}.pt"
        state = torch.load(model_path, map_location=self.device)
        model_string = 'model_' + str(i)
        model.load_state_dict(state[model_string])
        return model

    def predict(self, X_test, mask=None, test_idx=None, predict_raw=False):
        """Get predictions from the trained model.

        Parameters
        ----------
        targetgenes: array optional
            genes to be imputed
        Returns
        -------
        imputed : DataFrame
            imputed gene expression

        """
        predictors = self.predictors
        targets = self.targets

        if mask is not None:
            X_test, _, _ = self.maskdata(X_test, mask, test_idx)
        X_test_list = []
        for j, inputgenes in enumerate(predictors):
            X_test_list.append(X_test[:, inputgenes])

        # Make predictions using each subnetwork
        Y_pred_lst = []
        for i, model in enumerate(self.models):
            model = self.load_model(model, i)
            model.eval()
            with torch.no_grad():
                Y_pred_lst.append(model.forward(X_test_list[i].to(self.device)))
        # Concatenate predicted values
        Y_pred = torch.cat(Y_pred_lst, 1)
        gene_order = np.concatenate(targets)
        Y_pred = Y_pred[:, gene_order]

        # Convert back to counts
        if predict_raw:
            Y_pred = torch.expm1(Y_pred)

        return Y_pred

    def score(self, true_expr, imputed_expr, mask=None, metric="MSE", test_idx=None):
        """Scoring function of model.

        Parameters
        ----------
        true_expr :
            True underlying expression values
        imputed_expr :
            Imputed expression values
        test_idx :
            index of testing cells
        metric :
            Choice of scoring metric - 'RMSE' or 'ARI'

        Returns
        -------
        score :
            evaluation score

        """
        allowd_metrics = {"RMSE", "PCC"}
        if metric not in allowd_metrics:
            raise ValueError("scoring metric %r." % allowd_metrics)

        if test_idx is None:
            test_idx = range(len(true_expr))
        true_target = true_expr.to(self.device)
        imputed_target = imputed_expr.to(self.device)
        if mask is not None:  # and metric == 'MSE':
            # true_target = true_target[~mask[test_idx]]
            # imputed_target = imputed_target[~mask[test_idx]]
            imputed_target[mask[test_idx]] = true_target[mask[test_idx]]
        if metric == 'RMSE':
            return np.sqrt(F.mse_loss(true_target, imputed_target).item())
        elif metric == 'PCC':
            corr_cells = np.corrcoef(true_target.cpu(), imputed_target.cpu())
            return corr_cells
