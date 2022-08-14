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


def cell_topic_profile(X, groups, axis=1, method='median'):
    """cell_topic_profile.

    Parameters
    ----------
    X : torch 2-d tensor
        gene expression matrix, genes (rows) by sample cells (cols).
    groups : int
        cell-type labels of each sample cell in X.
    method : string optional
         method for reduction of cell-types for cell profile, default median.

    Returns
        -------
    X_profile : torch 2-d tensor
         cell profile matrix from scRNA-seq reference.

    """
    groups = np.array(groups)
    ids = np.unique(groups)
    if method == "median":
        X_profile = torch.Tensor(np.array([np.median(X[:, groups == ids[i]], axis=1) for i in range(len(ids))]).T)
    else:
        X_profile = torch.Tensor(np.array([np.mean(X[:, groups == ids[i]], axis=1) for i in range(len(ids))]).T)
    return X_profile


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
    rank : int
           rank of the matrix factorization.
    bias : boolean optional
           set true to include bias term in the regression modules, default False.
    profile_mtd : string optional
           method for reduction of cell-types for cell profile, default median.
    init : string optional
           initialization method for matrix factorization solver (see NMF from sklearn).
    max_iter : int optional
           maximum iterations allowed for matrix factorization solver.

    Returns
        -------
    None.

    """

    def __init__(self, in_dim, hid_dim, out_dim, rank, bias=False, profile_mtd='median', init='random', random_state=0,
                 max_iter=1000, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.bias = bias
        self.profile_mtd = 'median'

        self.nmf_model = NMF(Vshape=in_dim, rank=rank).to(device)
        #self.nmf_model = NMF(n_components=rank, init=init,  random_state=random_state, max_iter=max_iter)

        #self.nnls_reg1=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg1 = NNLS(in_dim=rank, out_dim=out_dim, bias=bias, device=device)

        #self.nnls_reg2=LinearRegression(fit_intercept=bias,positive=True)
        self.nnls_reg2 = NNLS(in_dim=hid_dim, out_dim=out_dim, bias=bias, device=device)

        self.model = nn.Sequential(self.nmf_model, self.nnls_reg1, self.nnls_reg2)

    def forward(self, x, y, cell_types):

        #1. get NMF decompositons
        W = self.nmf_model.H.clone()
        H = self.nmf_model.W.clone().T

        #2. get cell-topic and mix-topic profiles
        #a. get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        H_profile = cell_topic_profile(H.cpu().numpy(), groups=cell_types, axis=1, method=self.profile_mtd)
        H_profile = H_profile.to(self.device)

        #b. get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        #nnls ran for each spot
        B = self.nnls_reg1.model.weight.detach().T

        #3.  get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        P = self.nnls_reg2.model.weight.detach().T

        return (W, H_profile, B, P)

    def fit(self, x, y, cell_types):
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

        x = Variable(torch.FloatTensor(x), requires_grad=True).to(self.device)
        y = Variable(torch.FloatTensor(y)).to(self.device)

        #1. run NMF on scRNA X
        self.nmf_model.fit(x)
        self.nmf_model.requires_grad_(False)

        self.W = self.nmf_model.H.clone()
        self.H = self.nmf_model.W.clone().T

        #2. get cell-topic and mix-topic profiles
        #a. get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        self.H_profile = cell_topic_profile(self.H.cpu().numpy(), groups=cell_types, axis=1, method=self.profile_mtd)
        self.H_profile = self.H_profile.to(self.device)

        #b. get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        #nnls ran for each spot
        self.nnls_reg1.fit(self.W, y, max_iter=1000, lr=.01)
        self.nnls_reg1.requires_grad_(False)
        self.B = self.nnls_reg1.model.weight.clone().T

        #3.  get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        self.nnls_reg2.fit(self.H_profile, self.B, max_iter=1000, lr=.01)
        self.nnls_reg2.requires_grad_(False)
        self.P = self.nnls_reg2.model.weight.clone().T

    def predict(self, x, y, cell_types):
        """prediction function.
        Parameters
        ----------
        y : torch 2-d tensor
            mixed cell expression.

        Returns
        -------
        P : torch 2-d tensor
            predictions of cell-type proportions.

        """

        x = torch.FloatTensor(x).to(self.device)
        y = torch.FloatTensor(y).to(self.device)

        W, H_profile, B, P = self.forward(x, y, cell_types=cell_types)
        return (P)

    def score(self, x, y, cell_types):
        """score.

        Parameters
        ----------
        x : torch 2-d tensor
            scRNA-seq reference expression.
        y : torch 2-d tensor
            mixed cell expression to be deconvoluted.
        cell_types : torch tensor
            cell type annotations for scRNA-seq reference.

        Returns
        -------
        model_score : float
            coefficient of determination of the prediction (final non-negative linear module).

        """
        x = torch.FloatTensor(x).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        W, H_profile, B, P = self.forward(x, y, cell_types=cell_types)
        B_pred = self.nnls_reg2(H_profile)

        criterion = nn.MSELoss()
        model_score = criterion(B_pred, B).item()
        return model_score
