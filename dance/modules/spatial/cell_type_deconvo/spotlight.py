"""PyTorch reimplementation of the SPOTlight cell-type deconvolution method.

Adaptded from https://github.com/MarcElosua/SPOTlight

Reference
---------
Elosua-Bayes, Nieto, Mereu, Gut, and Heyn H. "SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics spots with single-cell transcriptomes."
Nucleic Acids Research (2021)

"""

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchnmf.nmf import NMF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    return torch.Tensor(X_profile)


class NNLS(nn.Module):
    """NNLS.

    Parameters
    ----------
    in_dim : int
        input dimension.
    out_dim : int
        output dimension.
    bias : boolean optional
        include bias term, default False.

    Returns
    -------
    None.

    """

    def __init__(self, in_dim, out_dim, bias=False, init_bias=None,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        self.model.bias = init_bias
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

    def fit(self, x, y, max_iter, lr, print_res=False, print_period=100):
        """fit function for model training.

        Parameters
        ----------
        x :
            input.
        y :
            output.
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

        x = Variable(x, requires_grad=True)
        y = Variable(y)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for iteration in range(max_iter):
            iteration += 1
            y_pred = self.model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            self.loss = loss
            # Zero gradients, perform a backward pass,
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #force non-negative weights only
            with torch.no_grad():
                self.model.weight.copy_(self.model.weight.data.clamp(min=0))
            #if iteration % print_period  == 0:
            #    print(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")


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
    ct_varname : str, optional
        Name of the cell-types column.
    ct_select : str, optional
        Selected cell-types to be considered for deconvolution.
    rank : int optional
           rank of the matrix factorization.
    sc_profile: numpy array optional
        pre-constructed cell profile matrix.
    bias : boolean optional
        include bias term, default False.
    init_bias: numpy array optional
        initial bias term (background estimate).
    init : string optional
           initialization method for matrix factorization solver (see NMF from sklearn).
    max_iter : int optional
           maximum iterations allowed for matrix factorization solver.

    Returns
    -------
    None.

    """

    def __init__(self, sc_count, sc_annot, mix_count, ct_varname, ct_select, rank=2, sc_profile=None, bias=False,
                 init_bias=None, init='random', max_iter=1000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.bias = bias
        self.ct_select = ct_select
        #subset sc samples on selected cell types (mutual between sc and mix cell data)
        ct_select_ix = sc_annot[sc_annot[ct_varname].isin(ct_select)].index
        self.sc_annot = sc_annot.loc[ct_select_ix]
        self.sc_count = sc_count.loc[ct_select_ix]
        cellTypes = self.sc_annot[ct_varname].values.tolist()
        self.cellTypes = cellTypes
        #construct a cell profile matrix if not profided
        if sc_profile is None:
            self.ref_sc_profile = cell_topic_profile(self.sc_count.values, cellTypes, ct_select, method='median')
        else:
            self.ref_sc_profile = sc_profile

        self.mix_count = mix_count.values.T
        self.sc_count = self.sc_count.values.T

        in_dim = self.sc_count.shape
        hid_dim = len(ct_select)
        out_dim = self.mix_count.shape[1]
        self.nmf_model = NMF(Vshape=in_dim, rank=rank).to(device)
        if rank == len(ct_select):
            #initialize basis as cell profile
            self.nmf_model.H = nn.Parameter(torch.Tensor(self.ref_sc_profile))
        #self.nmf_model = NMF(n_components=rank, init=init,  random_state=random_state, max_iter=max_iter)

        #self.nnls_reg1=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg1 = NNLS(in_dim=rank, out_dim=out_dim, bias=bias, device=device)

        #self.nnls_reg2=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg2 = NNLS(in_dim=hid_dim, out_dim=out_dim, bias=bias, device=device)

        self.model = nn.Sequential(self.nmf_model, self.nnls_reg1, self.nnls_reg2)

    def forward(self):

        #1. get NMF decompositons
        W = self.nmf_model.H.clone()
        H = self.nmf_model.W.clone().T

        #2. get cell-topic and mix-topic profiles
        #a. get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        H_profile = cell_topic_profile(H.cpu().numpy().T, self.cellTypes, self.ct_select, method='median')
        H_profile = H_profile.to(self.device)

        #b. get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        #nnls ran for each spot
        B = self.nnls_reg1.model.weight.detach().T

        #3.  get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        P = self.nnls_reg2.model.weight.detach().T

        return (W, H_profile, B, P)

    def fit(self, lr, max_iter):
        """fit function for model training.

        Parameters
        ----------
        x :
            scRNA-seq reference expression.
        y :
            mixed cell expression to be deconvoluted.
        cell_types :
            cell type annotations for scRNA-seq reference.

        Returns
        -------
        None.

        """

        x = Variable(torch.FloatTensor(self.sc_count), requires_grad=True).to(self.device)
        y = Variable(torch.FloatTensor(self.mix_count)).to(self.device)

        #1. run NMF on scRNA X
        self.nmf_model.fit(x, max_iter=max_iter)
        self.nmf_model.requires_grad_(False)

        self.W = self.nmf_model.H.clone()
        self.H = self.nmf_model.W.clone().T

        #2. get cell-topic and mix-topic profiles
        #a. get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        self.H_profile = cell_topic_profile(self.H.cpu().numpy().T, self.cellTypes, self.ct_select, method='median')
        self.H_profile = self.H_profile.to(self.device)

        #b. get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        #nnls ran for each spot
        self.nnls_reg1.fit(self.W, y, max_iter=max_iter, lr=lr)
        self.nnls_reg1.requires_grad_(False)
        self.B = self.nnls_reg1.model.weight.clone().T

        #3.  get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        self.nnls_reg2.fit(self.H_profile, self.B, max_iter=max_iter, lr=lr)
        self.nnls_reg2.requires_grad_(False)
        self.P = self.nnls_reg2.model.weight.clone().T

    def predict(self):
        """prediction function.
        Parameters
        ----------

        None.

        Returns
        -------
        P : torch 2-d tensor
            predictions of cell-type proportions.

        """
        W, H_profile, B, P = self.forward()
        proportion_preds = P / torch.sum(P, axis=0, keepdims=True).clamp(min=1e-6)
        return (proportion_preds)

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
