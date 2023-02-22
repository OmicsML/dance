"""PyTorch reimplementation of the SPOTlight cell-type deconvolution method.

Adaptded from https://github.com/MarcElosua/SPOTlight

Reference
---------
Elosua-Bayes, Nieto, Mereu, Gut, and Heyn H. "SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics
spots with single-cell transcriptomes." Nucleic Acids Research (2021)

"""
import torch
from torch import nn, optim
from torchnmf.nmf import NMF

from dance.transforms.pseudo_gen import get_ct_profile
from dance.utils import get_device
from dance.utils.wrappers import CastOutputType

get_ct_profile_tensor = CastOutputType(torch.FloatTensor)(get_ct_profile)


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

    def __init__(self, in_dim, out_dim, bias=False, init_bias=None, device="auto"):
        super().__init__()
        self.device = get_device(device)
        self.model = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        self.model.bias = init_bias
        self.model = self.model.to(self.device)

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
    """SPOTlight.

    Parameters
    ----------
    ref_count : pd.DataFrame
        Reference single cell RNA-seq counts data (cell x gene).
    ref_annot : pd.DataFrame
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

    """

    def __init__(self, ct_select, rank=2, sc_profile=None, bias=False, init_bias=None, init="random", device="auto"):
        super().__init__()
        self.device = get_device(device)
        self.bias = bias
        self.ct_select = ct_select
        self.rank = rank

    def _init_model(self, dim_out, ref_count, ref_annot):
        hid_dim = len(self.ct_select)
        self.nmf_model = NMF(Vshape=ref_count.T.shape, rank=self.rank).to(self.device)
        if self.rank == len(self.ct_select):  # initialize basis as cell profile
            self.nmf_model.H = nn.Parameter(get_ct_profile_tensor(ref_count, ref_annot, self.ct_select))

        self.nnls_reg1 = NNLS(in_dim=self.rank, out_dim=dim_out, bias=self.bias, device=self.device)
        self.nnls_reg2 = NNLS(in_dim=hid_dim, out_dim=dim_out, bias=self.bias, device=self.device)

        self.model = nn.Sequential(self.nmf_model, self.nnls_reg1, self.nnls_reg2)

    def forward(self, ref_annot):
        # Get NMF decompositions
        W = self.nmf_model.H.clone()
        H = self.nmf_model.W.clone().T

        # Get cell-topic and mix-topic profiles
        # Get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        H_profile = get_ct_profile_tensor(H.cpu().numpy().T, ref_annot, self.ct_select)
        H_profile = H_profile.to(self.device)

        # Get mix-topic profiles B: NNLS of basis W onto mix expression Y -- y ~ W*b
        # nnls ran for each spot
        B = self.nnls_reg1.model.weight.detach().T

        # Get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        P = self.nnls_reg2.model.weight.detach().T

        return (W, H_profile, B, P)

    def fit(self, x, ref_count, ref_annot, lr=1e-3, max_iter=1000):
        """Fit function for model training.

        Parameters
        ----------
        x :
            Mixed cell expression to be deconvoluted (cell x gene).
        lr : float
            Learning rate.
        max_iter : int
            Maximum iterations allowed for matrix factorization solver.

        """
        self._init_model(x.shape[0], ref_count, ref_annot)
        x = torch.FloatTensor(x.T).to(self.device)
        y = torch.FloatTensor(ref_count.T).to(self.device)

        # Run NMF on scRNA X
        self.nmf_model.fit(y, max_iter=max_iter)
        self.nmf_model.requires_grad_(False)

        self.W = self.nmf_model.H.clone()
        self.H = self.nmf_model.W.clone().T

        # Get cell-topic and mix-topic profiles
        # Get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        self.H_profile = get_ct_profile_tensor(self.H.cpu().numpy().T, ref_annot, self.ct_select)
        self.H_profile = self.H_profile.to(self.device)

        # Get mix-topic profiles B: NNLS of basis W onto mix expression X ~ W*b
        # nnls ran for each spot
        self.nnls_reg1.fit(self.W, x, max_iter=max_iter, lr=lr)
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
            Predicted cell-type proportions (cell x cell-type).

        """
        pred = self.P / torch.sum(self.P, axis=0, keepdims=True).clamp(min=1e-6)
        return pred.T

    def fit_and_predict(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        pred = self.predict()
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
