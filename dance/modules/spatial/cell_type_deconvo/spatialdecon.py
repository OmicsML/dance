"""A PyTorch reimplementation of the SpatialDecon cell-type deconvolution method.

Adapted from https: https://github.com/Nanostring-Biostats/SpatialDecon

Reference
---------
Danaher, Kim, Nelson, et al. "Advances in mixed cell deconvolution enable quantification of cell types in spatial transcriptomic data."
Nature Communications (2022)

"""
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


class SpatialDecon:
    """SpatialDecon.

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
        proportion_preds = proportion_preds / torch.sum(proportion_preds, axis=0)
        return (proportion_preds)

    def fit(self, ref_x, y, max_iter, lr, print_res=False, print_period=100):
        """fit function for model training.

        Parameters
        ----------
        ref_x :
            scRNA-seq reference expression.
        y :
            mixed cell expression.
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

        ref_x = Variable(torch.FloatTensor(ref_x), requires_grad=True).to(self.device)
        y = Variable(torch.FloatTensor(y)).to(self.device)

        criterion = MSLELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for iteration in range(max_iter):
            iteration += 1
            y_pred = self.model(ref_x)

            # Compute and print loss
            loss = criterion(y_pred, y)
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

    def score(self, ref_x, y):
        """score.

        Parameters
        ----------
        ref_x :
            scRNA reference expression
        y :
            cell-mixture expression

        Returns
        -------
        model_score : float
            MSLE loss between transformed scRNA reference expression (prediction) and cell-mixture expression.

        """
        ref_x = torch.FloatTensor(ref_x).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        y_pred = self.model(ref_x)

        criterion = MSLELoss()
        model_score = criterion(y_pred, y).item()
        return model_score
