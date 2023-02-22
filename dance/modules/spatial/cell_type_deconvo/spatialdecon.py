"""A PyTorch reimplementation of the SpatialDecon cell-type deconvolution method.

Adapted from https: https://github.com/Nanostring-Biostats/SpatialDecon

Reference
---------
Danaher, Kim, Nelson, et al. "Advances in mixed cell deconvolution enable quantification of cell types in spatial
transcriptomic data." Nature Communications (2022)

"""
import torch
import torch.nn as nn
from torch import optim

from dance.utils import get_device
from dance.utils.matrix import normalize


class MSLELoss(nn.Module):
    """Mean squared log error loss."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, true):
        """Forward function.

        Parameters
        ----------
        pred : torch.Tensor
            Linear transformation of cell profile (reference basis) matrix.
        true : torch.Tensor
            Mixture expression matrix.

        Returns
        -------
        loss : float
            Mean squared log error loss.

        """
        loss = self.mse(pred.clip(0).log1p(), true.clip(0).log1p())
        return loss


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
        Pre-constructed cell profile matrix.
    bias : boolean optional
        Include bias term, default False.
    init_bias: numpy array optional
        Initial bias term (background estimate).

    """

    def __init__(self, ct_select, sc_profile=None, bias=False, init_bias=None, device="auto"):
        super().__init__()
        self.device = get_device(device)
        self.ct_select = ct_select
        self.bias = bias
        self.init_bias = init_bias
        self.model = None

    def _init_model(self, num_cells: int, bias: bool = True):
        num_cell_types = len(self.ct_select)
        model = nn.Linear(in_features=num_cell_types, out_features=num_cells, bias=self.bias)
        if self.init_bias is not None:
            model.bias = nn.Parameter(torch.Tensor(self.init_bias.values.T.copy()))
        self.model = model.to(self.device)

    def forward(self, x: torch.Tensor):
        out = self.model(x)
        return (out)

    def predict(self):
        """Return fiited parameters as cell-type portion predictions.

        Returns
        -------
        proportion_preds : torch.Tensor
            Predictions of cell-type proportions (cell x cell-type).

        """
        weights = self.model.weight.clone().detach().cpu()
        proportion_preds = normalize(weights, mode="normalize", axis=1)
        return proportion_preds

    def fit(self, x, ct_profile, lr=1e-4, max_iter=500, print_res=False, print_period=100):
        """fit function for model training.

        Parameters
        ----------
        x
            Input expression matrix (cell x gene).
        max_iter : int
            Maximum number of iterations for optimizat.
        lr : float
            Learning rate.
        print_res : bool optional
            Indicates to print live training results, default False.
        print_period : int optional
            Indicates number of iterations until training results print.

        """
        ref_ct_profile = ct_profile.to(self.device)
        mix_count = x.T.to(self.device)
        self._init_model(x.shape[0])

        criterion = MSLELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for iteration in range(max_iter):
            iteration += 1
            mix_pred = self.model(ref_ct_profile)

            loss = criterion(mix_pred, mix_count)
            self.loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.model.weight.copy_(self.model.weight.data.clamp(min=0))
            if iteration % print_period == 0:
                print(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")

    def fit_and_predict(self, *args, **kwargs):
        """Fit parameters and return cell-type portion predictions."""
        self.fit(*args, **kwargs)
        pred = self.predict()
        return pred

    def score(self, pred, true):
        """Evaluate predictions.

        Parameters
        ----------
        pred
            Predicted cell-type proportions.
        true
            True cell-type proportions.

        Returns
        -------
        loss : float
            MSE loss between predicted and true cell-type proportions.

        """
        true = torch.FloatTensor(true).to(self.device)
        pred = torch.FloatTensor(pred).to(self.device)
        loss = nn.MSELoss()(pred, true)
        return loss.item()
