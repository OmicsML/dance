"""PyTorch reimplementation of the SPOTlight cell-type deconvolution method.

Adaptded from https://github.com/MarcElosua/SPOTlight

Reference
---------
Elosua-Bayes, Nieto, Mereu, Gut, and Heyn H. "SPOTlight: seeded NMF regression to deconvolute spatial transcriptomics
spots with single-cell transcriptomes." Nucleic Acids Research (2021)

"""
from functools import partial

import numpy as np
import torch
from torch import nn, optim
from torchnmf.nmf import NMF

from dance import logger
from dance.modules.base import BaseRegressionMethod
from dance.transforms import SetConfig
from dance.transforms.pseudo_gen import get_ct_profile
from dance.typing import Any, List, LogLevel, Optional
from dance.utils import get_device
from dance.utils.wrappers import CastOutputType

get_ct_profile_tensor = CastOutputType(torch.FloatTensor)(partial(get_ct_profile, method="median"))


class NNLS(nn.Module):
    """NNLS.

    Parameters
    ----------
    in_dim
        Input dimension.
    out_dim
        Output dimension.
    bias
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
        x
            Input.
        y
            Output.
        max_iter
            Maximum number of iterations for optimizat.
        lr
            Learning rate.
        print_res
            Indicates to print live training results, default False.
        print_period
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
                logger.info(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")


class SPOTlight(BaseRegressionMethod):
    """SPOTlight.

    Parameters
    ----------
    ref_count
        Reference single cell RNA-seq counts data (cell x gene).
    ref_annot
        Reference cell-type label information.
    ct_select
        Selected cell-types to be considered for deconvolution.
    rank
        Rank of the matrix factorization.
    bias
        Include bias term, default False.
    init_bias
        Initial bias term (background estimate).

    """

    def __init__(
        self,
        ref_count: np.ndarray,
        ref_annot: np.ndarray,
        ct_select: List[str],
        rank: int = 2,
        bias=False,
        init_bias=None,
        device="auto",
    ):
        self.ref_count = ref_count
        self.ref_annot = ref_annot
        self.ct_select = ct_select

        self.bias = bias
        self.rank = rank

        self.device = get_device(device)

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        return SetConfig({"label_channel": "cell_type_portion"}, log_level=log_level)

    def _init_model(self, dim_out, ref_count, ref_annot):
        hid_dim = len(self.ct_select)
        self.nmf_model = NMF(Vshape=ref_count.T.shape, rank=self.rank).to(self.device)
        if self.rank == len(self.ct_select):  # initialize basis as cell profile
            self.nmf_model.H = nn.Parameter(get_ct_profile_tensor(ref_count, ref_annot, ct_select=self.ct_select))

        self.nnls_reg1 = NNLS(in_dim=self.rank, out_dim=dim_out, bias=self.bias, device=self.device)
        self.nnls_reg2 = NNLS(in_dim=hid_dim, out_dim=dim_out, bias=self.bias, device=self.device)

        self.model = nn.Sequential(self.nmf_model, self.nnls_reg1, self.nnls_reg2)

    def fit(
        self,
        x: torch.Tensor,
        lr: float = 1e-3,
        max_iter: int = 1000,
    ):
        """Fit function for model training.

        Parameters
        ----------
        x
            Mixed cell expression to be deconvoluted (cell x gene).
        lr
            Learning rate.
        max_iter
            Maximum iterations allowed for matrix factorization solver.

        """
        ref_annot = self.ref_annot
        ct_select = self.ct_select
        device = self.device

        self._init_model(x.shape[0], self.ref_count, ref_annot)
        x = x.T.to(device)
        x_ref = torch.FloatTensor(self.ref_count.T).to(device)

        # Run NMF on scRNA X
        self.nmf_model.fit(x_ref, max_iter=max_iter)
        self.nmf_model.requires_grad_(False)
        self.W = self.nmf_model.H.clone()
        self.H = self.nmf_model.W.clone().T

        # Get cell-topic profiles H_profile: cell-type group medians of coef H (topic x cells)
        self.H_profile = get_ct_profile_tensor(self.H.cpu().numpy().T, ref_annot, ct_select=ct_select).to(device)

        # Get mix-topic profiles B: NNLS of basis W onto mix expression X ~ W*b
        # nnls ran for each spot
        self.nnls_reg1.fit(self.W, x, max_iter=max_iter, lr=lr)
        self.nnls_reg1.requires_grad_(False)
        self.B = self.nnls_reg1.model.weight.clone().T

        # Get cell-type proportions P: NNLS of cell-topic profile H_profile onoto mix-topic profile B -- b ~ h_profile*p
        self.nnls_reg2.fit(self.H_profile, self.B, max_iter=max_iter, lr=lr)
        self.nnls_reg2.requires_grad_(False)
        self.P = self.nnls_reg2.model.weight.clone().T

    def predict(self, x: Optional[Any] = None):
        """Prediction function.

        Parameters
        ----------
        x
            Not used, for compatibility with the BaseRegressionMethod class.

        Returns
        -------
        pred
            Predicted cell-type proportions (cell x cell-type).

        """
        return nn.functional.normalize(self.P.detach().clone().cpu().T, dim=1, p=1)
