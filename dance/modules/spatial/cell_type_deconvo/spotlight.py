"""PyTorch reimplementation of the SPOTlight cell-type deconvolution method.

Adaptded from https://github.com/MarcElosua/SPOTlight

Reference
---------
Elosua-Bayes, Nieto, Mereu, Gut, and Heyn H. "SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics
spots with single-cell transcriptomes." Nucleic Acids Research (2021)

"""

import numpy as np
import torch
from torch import nn, optim
from torchnmf.nmf import NMF


def cell_topic_profile(x, groups, ct_select, axis=0, method="median"):
    """Cell topic profile.

    Parameters
    ----------
    x : torch.Tensor
        Gene expression matrix (gene x cells).
    groups : int
        Cell-type labels of each sample cell in x.
    ct_select:
        Cell-types to profile.
    method : str
         Method for reduction of cell-types for cell profile, default median.

    Returns
    -------
    x_profile : torch 2-d tensor
         cell profile matrix from scRNA-seq reference.

    """
    if method == "median":
        x_profile = np.array([
            np.median(x[[i for i in range(len(groups)) if groups[i] == ct_select[j]], :], axis=0)
            for j in range(len(ct_select))
        ]).T
    else:
        x_profile = np.array([
            np.mean(x[[i for i in range(len(groups)) if groups[i] == ct_select[j]], :], axis=0)
            for j in range(len(ct_select))
        ]).T
    return torch.Tensor(x_profile)


class NNLS(nn.Module):
    """NNLS.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    out_dim : int
        Output dimension.
    bias : bool
        Include bias term, default False.

    """

    def __init__(self, in_dim, out_dim, bias=False, init_bias=None, device="cpu"):
        super().__init__()
        self.device = device
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        self.model.bias = init_bias
        self.model = self.model.to(device)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return (out)

    def fit(self, x, y, max_iter, lr, print_res=False, print_period=100):
        """Fit function for model training.

        Parameters
        ----------
        x :
            Input.
        y :
            Output.
        max_iter : int
            Maximum number of iterations for optimizat.
        lr : float
            Learning rate.
        print_res : bool
            Indicates to print live training results, default False.
        print_period : int
            Indicates number of iterations until training results print.

        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for iteration in range(max_iter):
            iteration += 1
            y_pred = self.model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Force non-negative weights only
            with torch.no_grad():
                self.model.weight.copy_(self.model.weight.data.clamp(min=0))

            if iteration % print_period == 0:
                print(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")


class SPOTlight:
    """SPOTlight class.

    Parameters
    ----------
    sc_count : pd.DataFrame
        Reference single cell RNA-seq counts data.
    sc_annot : pd.DataFrame
        Reference cell-type label information.
    mix_count : pd.DataFrame
        Target mixed-cell RNA-seq counts data to be deconvoluted.
    ct_varname : str
        Name of the cell-types column.
    ct_select : str
        Selected cell-types to be considered for deconvolution.
    rank : int
        Rank of the matrix factorization.
    sc_profile: np.ndarray
        Pre-constructed cell profile matrix.
    bias : bool
        Include bias term, default False.
    init_bias: np.ndarray
        Initial bias term (background estimate).
    init : str
        Initialization method for matrix factorization solver (see NMF from sklearn).
    max_iter : int
        Maximum iterations allowed for matrix factorization solver.

    """

    def __init__(self, sc_count, sc_annot, mix_count, ct_varname, ct_select, rank=2, sc_profile=None, bias=False,
                 init_bias=None, init="random", max_iter=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.bias = bias
        self.ct_select = ct_select

        # Subset sc samples on selected cell types (mutual between sc and mix cell data)
        ct_select_ix = sc_annot[sc_annot[ct_varname].isin(ct_select)].index
        self.sc_annot = sc_annot.loc[ct_select_ix]
        self.sc_count = sc_count.loc[ct_select_ix]
        cellTypes = self.sc_annot[ct_varname].values.tolist()
        self.cellTypes = cellTypes

        # Construct a cell profile matrix if not profided
        if sc_profile is None:
            self.ref_sc_profile = cell_topic_profile(self.sc_count.values, cellTypes, ct_select, method="median")
        else:
            self.ref_sc_profile = sc_profile

        self.mix_count = mix_count.values.T
        self.sc_count = self.sc_count.values.T

        in_dim = self.sc_count.shape
        hid_dim = len(ct_select)
        out_dim = self.mix_count.shape[1]
        self.nmf_model = NMF(Vshape=in_dim, rank=rank).to(device)
        if rank == len(ct_select):  # initialize basis as cell profile
            self.nmf_model.H = nn.Parameter(torch.Tensor(self.ref_sc_profile))

        # self.nmf_model = NMF(n_components=rank, init=init,  random_state=random_state, max_iter=max_iter)
        # self.nnls_reg1=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg1 = NNLS(in_dim=rank, out_dim=out_dim, bias=bias, device=device)

        # self.nnls_reg2=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg2 = NNLS(in_dim=hid_dim, out_dim=out_dim, bias=bias, device=device)

        self.model = nn.Sequential(self.nmf_model, self.nnls_reg1, self.nnls_reg2)

    def forward(self):
        # Get NMF decompositons
        W = self.nmf_model.H.clone()
        H = self.nmf_model.W.clone().T

        # Get cell-topic and mix-topic profiles
        # Get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        H_profile = cell_topic_profile(H.cpu().numpy().T, self.cellTypes, self.ct_select, method="median")
        H_profile = H_profile.to(self.device)

        # Get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        # nnls ran for each spot
        B = self.nnls_reg1.model.weight.detach().T

        # Get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        P = self.nnls_reg2.model.weight.detach().T

        return (W, H_profile, B, P)

    def fit(self, lr, max_iter):
        """Fit function for model training.

        Parameters
        ----------
        x :
            scRNA-seq reference expression.
        y :
            Mixed cell expression to be deconvoluted.
        cell_types :
            Cell-type annotations for scRNA-seq reference.

        """

        x = torch.FloatTensor(self.sc_count).to(self.device)
        y = torch.FloatTensor(self.mix_count).to(self.device)

        # Run NMF on scRNA X
        self.nmf_model.fit(x, max_iter=max_iter)
        self.nmf_model.requires_grad_(False)

        self.W = self.nmf_model.H.clone()
        self.H = self.nmf_model.W.clone().T

        # Get cell-topic and mix-topic profiles
        # Get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        self.H_profile = cell_topic_profile(self.H.cpu().numpy().T, self.cellTypes, self.ct_select, method="median")
        self.H_profile = self.H_profile.to(self.device)

        # Get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        # nnls ran for each spot
        self.nnls_reg1.fit(self.W, y, max_iter=max_iter, lr=lr)
        self.nnls_reg1.requires_grad_(False)
        self.B = self.nnls_reg1.model.weight.clone().T

        # Get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        self.nnls_reg2.fit(self.H_profile, self.B, max_iter=max_iter, lr=lr)
        self.nnls_reg2.requires_grad_(False)
        self.P = self.nnls_reg2.model.weight.clone().T

    def predict(self):
        """Prediction function.

        Returns
        -------
        pred : torch.Tensor
            Predicted cell-type proportions.

        """
        W, H_profile, B, P = self.forward()
        pred = P / torch.sum(P, axis=0, keepdims=True).clamp(min=1e-6)
        return pred

    def score(self, pred, true):
        """Score function.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted cell-type proportions.
        true : torch.Tensor
            True cell-type proportions.

        Returns
        -------
        loss : float
            MSE loss between predicted and true cell-type proportions.

        """
        pred = pred.to(self.device)
        true = true.to(self.device)
        loss = nn.functional.mse_loss(pred, true)
        return loss.item()
