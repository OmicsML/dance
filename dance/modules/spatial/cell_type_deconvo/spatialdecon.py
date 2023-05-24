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

from dance import logger
from dance.modules.base import BaseRegressionMethod
from dance.transforms import CellTopicProfile, Compose, SetConfig
from dance.typing import Any, LogLevel, Optional
from dance.utils import get_device


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


class SpatialDecon(BaseRegressionMethod):
    """SpatialDecon.

    Parameters
    ----------
    ct_profile
        Cell type characteristic profiles (cell-type x gene).
    ct_select
        Selected cell-types to be considered for deconvolution.
    bias
        Include bias term, default False.

    """

    def __init__(self, ct_profile, ct_select, bias=False, device="auto"):
        self.ct_profile = ct_profile
        self.ct_select = ct_select
        self.bias = bias
        self.device = get_device(device)

    @staticmethod
    def preprocessing_pipeline(ct_select: str = "auto", ct_profile_split: str = "ref", log_level: LogLevel = "INFO"):
        return Compose(
            CellTopicProfile(ct_select=ct_select, split_name="ref"),
            SetConfig({"label_channel": "cell_type_portion"}),
            log_level=log_level,
        )

    def _init_model(self, num_cells: int, bias: bool = True):
        num_cell_types = len(self.ct_select)
        model = nn.Linear(in_features=num_cell_types, out_features=num_cells, bias=self.bias)
        self.model = model.to(self.device)

    def predict(self, x: Optional[Any] = None):
        """Return fiited parameters as cell-type portion predictions.

        Parameters
        ----------
        x
            Not used, for compatibility with the BaseRegressionMethod class.

        Returns
        -------
        proportion_preds
            Predictions of cell-type proportions (cell x cell-type).

        """
        weights = self.model.weight.clone().detach().cpu()
        return nn.functional.normalize(weights, dim=1, p=1)

    def fit(
        self,
        x: torch.Tensor,
        lr: float = 1e-4,
        max_iter: int = 500,
        print_period: int = 100,
    ):
        """Fit function for model training.

        Parameters
        ----------
        x
            Input expression matrix (cell x gene).
        lr
            Learning rate.
        max_iter
            Maximum number of iterations for optimizat.
        print_period
            Indicates number of iterations until training results print.

        """
        ref_ct_profile = self.ct_profile.to(self.device)
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
                logger.info(f"Epoch: {iteration:02}/{max_iter} Loss: {loss.item():.5e}")
