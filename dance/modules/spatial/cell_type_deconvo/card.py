"""Reimplementation of the CARD cell-type deconvolution method.

Extended from https://github.com/YingMa0107/CARD

Reference
---------
Ma, Ying, and Xiang Zhou. "Spatially informed cell-type deconvolution for spatial transcriptomics."
Nature Biotechnology (2022): 1-11.

"""
import numpy as np
import pandas as pd

from dance import logger
from dance.modules.base import BaseRegressionMethod
from dance.transforms import (CellTopicProfile, Compose, FilterGenesCommon, FilterGenesMarker, FilterGenesMatch,
                              FilterGenesPercentile, SetConfig)
from dance.typing import Any, LogLevel, Optional, Tuple
from dance.utils.matrix import normalize, pairwise_distance


def obj_func(trac_xxt, UtXV, UtU, VtV, mGene, nSample, b, Lambda, beta, vecOne, V, L, alpha, sigma_e2=None):
    normNMF = trac_xxt - 2.0 * np.trace(UtXV) + np.trace(UtU @ VtV)
    sigma_e2 = normNMF / (mGene * nSample) or sigma_e2
    logX = -(mGene * nSample) * 0.5 * np.log(sigma_e2) - 0.5 * (normNMF / sigma_e2)
    temp = (V.T - b @ vecOne.T) @ L @ (V - vecOne @ b.T)
    logV = -(nSample) * 0.5 * np.sum(np.log(Lambda)) - 0.5 * (np.sum(np.diag(temp) / Lambda))
    logSigmaL2 = -(alpha + 1.0) * np.sum(np.log(Lambda)) - np.sum(beta / Lambda)
    logger.debug(f"{logX=:5.2e}, {logV=:5.2e}, {logSigmaL2=:5.2e}")
    return logX + logV + logSigmaL2


def CARDref(Xinput, U, W, phi, max_iter, epsilon, V, b, sigma_e2, Lambda):
    # Make necessary deep copies
    V = V.copy()

    # Initialize some useful items
    nSample = int(Xinput.shape[1])  # number of spatial sample points
    mGene = int(Xinput.shape[0])  # number of genes in spatial deconvolution
    k = int(U.shape[1])  # number of cell type
    L = np.zeros((nSample, nSample))
    D = np.zeros((nSample, nSample))
    V_old = np.zeros((nSample, k))
    UtU = np.zeros((k, k))
    VtV = np.zeros((k, k))
    colsum_W = np.zeros((nSample, 1))
    UtX = np.zeros((k, nSample))
    XtU = np.zeros((nSample, k))
    UtXV = np.zeros((k, k))
    temp = np.zeros((k, k))
    part1 = np.zeros((nSample, k))
    part2 = np.zeros((nSample, k))
    updateV_k = np.zeros(k)
    updateV_den_k = np.zeros(k)
    vecOne = np.ones((nSample, 1))
    diag_UtU = np.zeros(k)
    alpha = 1.0
    beta = nSample / 2.0
    accu_L = 0.0
    trac_xxt = (Xinput * Xinput).sum()

    # Initialize values with constant matrix calculations for increasing speed
    UtX = U.T @ Xinput
    XtU = UtX.T
    UtXV = UtX @ V
    VtV = V.T @ V
    UtU = U.T @ U
    diag_UtU = np.diag(UtU)
    if W is not None:
        colsum_W = np.sum(W, axis=1)
        D = np.diag(colsum_W)  # diagonal matrix whose entries are column
        L = D - phi * W  # graph laplacian
        colsum_W = colsum_W.reshape(nSample, 1)
        accu_L = np.sum(L)

    # Iteration starts
    obj = obj_func(trac_xxt, UtXV, UtU, VtV, mGene, nSample, b, Lambda, beta, vecOne, V, L, alpha, sigma_e2)
    for i in range(max_iter):
        obj_old = obj
        V_old = V.copy()

        Lambda = (np.diag(temp) / 2.0 + beta) / (nSample / 2.0 + alpha + 1.0)
        if W is not None:
            b = np.sum(V.T @ L, axis=1, keepdims=True) / accu_L
            part1 = sigma_e2 * (D @ V + phi * colsum_W @ b.T)
            part2 = sigma_e2 * (phi * W @ V + colsum_W @ b.T)

        for nCT in range(k):
            updateV_den_k = Lambda[nCT] * (V[:, nCT] * diag_UtU[nCT] +
                                           (V @ UtU[:, nCT] - V[:, nCT] * diag_UtU[nCT])) + part1[:, nCT]

            updateV_k = (Lambda[nCT] * XtU[:, nCT] + part2[:, nCT]) / updateV_den_k
            V[:, nCT] = V[:, nCT] * updateV_k

        UtXV = UtX @ V
        VtV = V.T @ V
        obj = obj_func(trac_xxt, UtXV, UtU, VtV, mGene, nSample, b, Lambda, beta, vecOne, V, L, alpha)

        logic1 = (obj > obj_old) & ((abs(obj - obj_old) * 2.0 / abs(obj + obj_old)) < epsilon)
        logic2 = np.sqrt(np.sum((V - V_old) * (V - V_old)) / (nSample * k)) < epsilon
        stop_logic = np.isnan(obj) or logic1 or logic2
        logger.debug(f"{i=:<4}, {obj=:.5e}")
        if stop_logic and i > 5:
            logger.info(f"Exiting at {i=}")
            break

    pred = V / V.sum(axis=1, keepdims=True)

    return pred, obj


class Card(BaseRegressionMethod):
    """The CARD cell-type deconvolution model.

    Parameters
    ----------
    basis
        The cell-type profile basis.

    """

    def __init__(self, basis: pd.DataFrame, random_state: Optional[int] = 42):
        self.basis = basis
        self.best_phi = None
        self.best_obj = -np.inf
        self.random_state = random_state

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        return Compose(
            CellTopicProfile(ct_select="auto", ct_key="cellType", batch_key=None, split_name="ref", method="mean"),
            FilterGenesMatch(prefixes=["mt-"], case_sensitive=False),
            FilterGenesCommon(split_keys=["ref", "test"]),
            FilterGenesMarker(ct_profile_channel="CellTopicProfile", threshold=1.25),
            FilterGenesPercentile(min_val=1, max_val=99, mode="rv"),
            SetConfig({
                "feature_channel": [None, "spatial"],
                "feature_channel_type": ["X", "obsm"],
                "label_channel": "cell_type_portion",
            }),
            log_level=log_level,
        )

    def fit(
        self,
        inputs: Tuple[np.ndarray, np.ndarray],
        y: Optional[Any] = None,
        max_iter: int = 100,
        epsilon: float = 1e-4,
        sigma: float = 0.1,
        location_free: bool = False,
    ):
        """Fit function for model training.

        Parameters
        ----------
        inputs
            A tuple containing (1) the input features encoding the scRNA-seq counts to be deconvoluted, and (2) a 2-d
            array of spatial location of each spot (spots x 2). If the spatial location information is all zero, or the
            ``location_free`` option is set to :obj:`True`, then do not use the spatial location information.
        y
            Not used, for compatibility with the BaseRegressionMethod class.
        max_iter
            Maximum number of iterations for optimization.
        epsilon
            Optimization threshold.
        sigma
            Spatial gaussian kernel scaling factor.
        location_free
            Do not use spatial location info if set to True.

        """
        x, spatial = inputs
        x_norm = normalize(x, axis=1, mode="normalize")

        # Spatial location
        if location_free or (spatial == 0).all():
            kernel_mat = None
        else:
            # TODO: refactor this to preprocess?
            norm_cords = (spatial - spatial.min(0))  # Q: why not min-max?
            norm_cords /= norm_cords.max()
            euclidean_distances = pairwise_distance(norm_cords.astype(np.float32), 0)
            kernel_mat = np.exp(-euclidean_distances**2 / (2 * sigma**2))
            np.fill_diagonal(kernel_mat, 0)

        # Scale the Xinput_norm and B to speed up the convergence.
        basis = self.basis.values.copy()
        x_norm = x_norm * 0.1 / x_norm.mean()
        b_mat = basis * 0.1 / basis.mean()

        # Initialize the proportion matrix
        rng = np.random.default_rng(self.random_state)
        Vint1 = rng.dirichlet(np.repeat(10, basis.shape[1], axis=0), x_norm.shape[0])
        phi = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]

        # Optimization
        for iphi in range(len(phi)):
            res, obj = CARDref(Xinput=x_norm.T, U=b_mat, W=kernel_mat, phi=phi[iphi], max_iter=max_iter,
                               epsilon=epsilon, V=Vint1, b=np.repeat(0, b_mat.shape[1]).reshape(b_mat.shape[1], 1),
                               sigma_e2=0.1, Lambda=np.repeat(10, basis.shape[1]))
            if obj > self.best_obj:
                self.res = res
                self.best_obj = obj
                self.best_phi = phi
        logger.info("Deconvolution finished")

    def predict(self, x: Optional[Any] = None) -> np.ndarray:
        """Prediction function.

        Parameters
        ----------
        x
            Not used, for compatibility with the BaseRegressionMethod class.

        Returns
        -------
        numpy.ndarray
            Predictions of cell-type proportions.

        """
        return self.res
