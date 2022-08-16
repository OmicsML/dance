"""A PyTorch reimplementation of the SpatialDecon cell-type deconvolution method.

Adapted from https: https://github.com/Nanostring-Biostats/SpatialDecon

Reference
---------
Danaher, Kim, Nelson, et al. "Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data."
Nature Communications (2022)

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSLELoss(nn.Module):
    """MSLELoss.

    Parameters
    ----------
    None

    Returns
    -------
    None.

    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        """forward function.

        Parameters
        ----------
        pred : torch tensor
            linear transformation of cell profile (reference basis) matrix.
        actual : torch tensor
            mixture expression matrix.

        Returns
        -------
        loss : float
            mean squared log error loss.

        """
        pred_clamp = pred.clamp(min=0)
        actual_clamp = actual.clamp(min=0)
        loss = self.mse(torch.log1p(pred_clamp), torch.log1p(actual_clamp))
        return loss


def cell_topic_profile(X, groups, ct_select, axis=0, method='median'):
    """cell_topic_profile.

    Parameters
    ----------
    X : torch 2-d tensor
        gene expression matrix, genes (rows) by sample cells (cols).
    groups : int
        cell-type labels of each sample cell in X.
    ct_select:
        cell types to profile.
    method : string optional
         method for reduction of cell-types for cell profile, default median.

    Returns
        -------
    X_profile : torch 2-d tensor
         cell profile matrix from scRNA-seq reference.

    """
    if method == "median":
        X_profile = np.array([
            np.median(X[[i for i in range(len(groups)) if groups[i] == ct_select[j]], :], axis=0)
            for j in range(len(ct_select))
        ]).T
    else:
        X_profile = np.array([
            np.mean(X[[i for i in range(len(groups)) if groups[i] == ct_select[j]], :], axis=0)
            for j in range(len(ct_select))
        ]).T
    return X_profile


class SpatialDecon:
    """SpatialDecon.

    Parameters
    ----------
    sc_count : pd.DataFrame
        Reference single cell RNA-seq counts data.
    sc_annot : pd.DataFrame
        Reference cell-type label information.
    mix_count : pd.DataFrame
        Target mixed-cell RNA-seq counts data to be deconvoluted.
    ct_varname : str, optional
        Name of the cell-types column.
    ct_select : str, optional
        Selected cell-types to be considered for deconvolution.
    sc_profile: numpy array optional
        pre-constructed cell profile matrix.
    bias : boolean optional
        include bias term, default False.
    init_bias: numpy array optional
        initial bias term (background estimate).

    Returns
    -------
    None.

    """

    def __init__(self, sc_count, sc_annot, mix_count, ct_varname, ct_select, sc_profile=None, bias=False,
                 init_bias=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        self.device = device

        #subset sc samples on selected cell types (mutual between sc and mix cell data)
        ct_select_ix = sc_annot[sc_annot[ct_varname].isin(ct_select)].index
        self.sc_annot = sc_annot.loc[ct_select_ix]
        self.sc_count = sc_count.loc[ct_select_ix]
        cellTypes = self.sc_annot[ct_varname].values.tolist()

        #construct a cell profile matrix if not profided
        if sc_profile is None:
            self.ref_sc_profile = cell_topic_profile(self.sc_count.values, cellTypes, ct_select, method='median')
        else:
            self.ref_sc_profile = sc_profile

        self.mix_count = mix_count.values.T

        in_dim = self.ref_sc_profile.shape[1]
        out_dim = self.mix_count.shape[1]
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        if init_bias is not None:
            self.model.bias = nn.Parameter(torch.Tensor(init_bias.values.T.copy()))
        self.model = self.model.to(device)

    def forward(self, x: torch.Tensor):
        """forward function.

        Parameters
        ----------
        x : torch tensor
            input features.

        Returns
        -------
        output : torch tensor
            linear projection of input.

        """
        out = self.model(x)
        return (out)

    def predict(self):
        """prediction function.
        Parameters
        ----------

        Returns
        -------

        proportion_preds : torch tensor
            predictions of cell-type proportions.

        """
        proportion_preds = self.model.weight.T
        proportion_preds = proportion_preds / torch.sum(proportion_preds, axis=0, keepdims=True).clamp(min=1e-6)
        return (proportion_preds)

    def fit(self, lr, max_iter, print_res=False, print_period=100):
        """fit function for model training.

        Parameters
        ----------
        max_iter : int
            maximum number of iterations for optimizat.
        lr : float
            learning rate.
        print_res : bool optional
            indicates to print live training results, default False.
        print_period : int optional
            indicates number of iterations until training results print.

        Returns
        -------
        None.

        """
        ref_sc_profile = Variable(torch.FloatTensor(self.ref_sc_profile), requires_grad=True).to(self.device)
        mix_count = Variable(torch.FloatTensor(self.mix_count)).to(self.device)

        criterion = MSLELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for iteration in range(max_iter):
            iteration += 1
            mix_pred = self.model(ref_sc_profile)

            # Compute and print loss
            loss = criterion(mix_pred, mix_count)
            self.loss = loss
            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.weight.copy_(self.model.weight.data.clamp(min=0))
            if iteration % print_period == 0:
                print(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")

    def score(self, pred, true_prop):
        """score.

        Parameters
        ----------
        pred :
            predicted cell-type proportions.
        true_prop :
            true cell-type proportions.

        Returns
        -------
        loss : float
            mse loss between predicted and true cell-type proportions.

        """
        true_prop = true_prop.to(self.device)
        pred = pred / torch.sum(pred, 1, keepdims=True).clamp(min=1e-6)
        true_prop = true_prop / torch.sum(true_prop, 1, keepdims=True).clamp(min=1e-6)
        loss = ((pred - true_prop)**2).mean()

        return loss.detach().item()
