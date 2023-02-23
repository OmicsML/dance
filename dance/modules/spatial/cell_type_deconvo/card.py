"""Reimplementation of the CARD cell-type deconvolution method.

Extended from https://github.com/YingMa0107/CARD

Reference
---------
Ma, Ying, and Xiang Zhou. "Spatially informed cell-type deconvolution for spatial transcriptomics."
Nature Biotechnology (2022): 1-11.

"""
import numpy as np
import pandas as pd

from dance.transforms import FilterGenesMarker
from dance.transforms.pseudo_gen import get_ct_profile
from dance.utils.matrix import pairwise_distance


def obj_func(trac_xxt, UtXV, UtU, VtV, mGene, nSample, b, Lambda, beta, vecOne, V, L, alpha, sigma_e2=None):
    normNMF = trac_xxt - 2.0 * np.trace(UtXV) + np.trace(UtU @ VtV)
    sigma_e2 = normNMF / (mGene * nSample) or sigma_e2
    logX = -(mGene * nSample) * 0.5 * np.log(sigma_e2) - 0.5 * (normNMF / sigma_e2)
    temp = (V.T - b @ vecOne.T) @ L @ (V - vecOne @ b.T)
    logV = -(nSample) * 0.5 * np.sum(np.log(Lambda)) - 0.5 * (np.sum(np.diag(temp) / Lambda))
    logSigmaL2 = -(alpha + 1.0) * np.sum(np.log(Lambda)) - np.sum(beta / Lambda)
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
    logicalLogL = False
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

    obj_old = obj_func(trac_xxt, UtXV, UtU, VtV, mGene, nSample, b, Lambda, beta, vecOne, V, L, alpha, sigma_e2)
    V_old = V.copy()

    # Iteration starts
    iter_converge = 0
    for i in range(max_iter):
        # logV = 0.0
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

        logicalLogL = (obj > obj_old) & ((abs(obj - obj_old) * 2.0 / abs(obj + obj_old)) < epsilon)
        # TODO: setup logging and make this debug or info
        # print(f"{i=:<4}, {obj=:.5e}, {logX=:5.2e}, {logV=:5.2e}, {logSigmaL2=:5.2e}")
        if (np.isnan(obj) | (np.sqrt(np.sum((V - V_old) * (V - V_old)) / (nSample * k)) < epsilon) | logicalLogL):
            if (i > 5):  # // run at least 5 iterations
                # print(f"Exiting at {i=}")
                iter_converge = i
                break
        else:
            obj_old = obj
            V_old = V.copy()

    res = {"V": V, "sigma_e2": sigma_e2, "Lambda": Lambda, "b": b, "Obj": obj, "iter_converge": iter_converge}
    return res


class Card:
    """The CARD cell-type deconvolution model.

    Parameters
    ----------
    sc_count : pd.DataFrame
        Reference single cell RNA-seq counts data.
    sc_meta : pd.DataFrame
        Reference cell-type label information.
    ct_varname : str, optional
        Name of the cell-types column.
    ct_select : str, optional
        Selected cell-types to be considered for deconvolution.
    sample_varname : str, optional
        Name of the samples column.
    minCountGene : int
        Minimum number of genes required.
    minCountSpot : int
        Minimum number of spots required.
    basis
        The basis parameter.
    markers
        Markers.

    """

    def __init__(self, sc_count, sc_meta, ct_varname=None, ct_select=None, sample_varname=None, minCountGene=100,
                 minCountSpot=5, basis=None, markers=None):
        self.sc_count = sc_count
        self.sc_meta = sc_meta
        self.ct_varname = ct_varname
        self.ct_select = ct_select
        self.sample_varname = sample_varname
        self.minCountGene = minCountGene
        self.minCountSpot = minCountSpot
        self.basis = basis
        self.marker = markers
        self.info_parameters = {}

        self.createscRef()  # create basis
        all_genes = sc_count.columns.tolist()
        gene_to_idx = {j: i for i, j in enumerate(all_genes)}
        not_mt_genes = [i for i in all_genes if not i.lower().startswith("mt-")]
        selected_genes = self.selectInfo(not_mt_genes)
        selected_gene_idx = list(map(gene_to_idx.get, selected_genes))
        self.gene_mask = np.zeros(len(all_genes), dtype=np.bool)
        self.gene_mask[selected_gene_idx] = True

    def createscRef(self):
        """CreatescRef - create reference basis matrix from reference scRNA-seq."""
        sc_meta = self.sc_meta.copy()
        ct_varname = self.ct_varname
        batch_varname = self.sample_varname
        if batch_varname is None:
            sc_meta["sampleID"] = "Sample"
            batch_varname = "sampleID"

        ct_profile = get_ct_profile(
            self.sc_count.values,
            sc_meta[ct_varname].values,
            ct_select=self.ct_select,
            batch_index=sc_meta[batch_varname].values,
        ).T
        basis = pd.DataFrame(ct_profile, index=self.ct_select, columns=self.sc_count.columns)
        self.basis = basis

    def selectInfo(self, common_gene):
        """Select Informative Genes used in the deconvolution.

        Parameters
        ----------
        common_gene : list
            Common genes between scRNAseq count data and spatial resolved transcriptomics data.

        Returns
        -------
        list
            List of informative genes.

        """
        # Select marker genes from common genes
        markers, _ = FilterGenesMarker.get_marker_genes(self.basis.values, self.basis.index.tolist(),
                                                        self.basis.columns.tolist())
        selected_genes = sorted(set(markers) & set(common_gene))

        # Compute coefficient of variation for each gene within each cell type
        counts = self.sc_count[selected_genes].join(self.sc_meta[self.ct_varname])
        cov_mean = (counts.groupby(self.ct_varname).var() / counts.groupby(self.ct_varname).mean()).mean()

        # Remove genes that have high cov
        ind = cov_mean < cov_mean.quantile(.99)
        genes = sorted(cov_mean[ind].index)

        return genes

    def fit(self, x, spatial, max_iter=100, epsilon=1e-4, sigma=0.1, location_free: bool = False):
        """Fit function for model training.

        Parameters
        ----------
        x : np.ndarray
            Target spatial RNA-seq counts data to be deconvoluted.
        spatial : np.ndarray
            2-D array of the spatial locations of each spot (spots x 2). If all zeros, then do not use spatial info.
        max_iter : int
            Maximum number of iterations for optimization.
        epsilon : float
            Optimization threshold.
        sigma : float
            Spatial gaussian kernel scaling factor.
        location_free
            Do not use spatial location info if set to True.

        """
        ct_select = self.ct_select
        Basis = self.basis.copy()
        Basis = Basis.loc[ct_select]

        gene_mask = self.gene_mask & (x.sum(0) > 0)
        B = Basis.values[:, gene_mask].copy()  # TODO: make it a numpy array
        Xinput = x[:, gene_mask].copy()
        Xinput_norm = Xinput / Xinput.sum(1, keepdims=True)  # TODO: use the normalize util

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

        # Initialize the proportion matrix
        rng = np.random.default_rng(20200107)
        Vint1 = rng.dirichlet(np.repeat(10, B.shape[0], axis=0), Xinput_norm.shape[0])
        phi = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        # scale the Xinput_norm and B to speed up the convergence.
        Xinput_norm = Xinput_norm * 0.1 / Xinput_norm.sum()
        B = B * 0.1 / B.mean()

        # Optimization
        ResList = {}
        Obj = np.array([])
        for iphi in range(len(phi)):
            res = CARDref(Xinput=Xinput_norm.T, U=B.T, W=kernel_mat, phi=phi[iphi], max_iter=max_iter, epsilon=epsilon,
                          V=Vint1, b=np.repeat(0, B.T.shape[1]).reshape(B.T.shape[1], 1), sigma_e2=0.1,
                          Lambda=np.repeat(10, len(ct_select)))
            ResList[str(iphi)] = res
            Obj = np.append(Obj, res["Obj"])
        self.Obj_hist = Obj
        Optimal_ix = np.where(Obj == Obj.max())[0][-1]  # in case if there are two equal objective function values
        OptimalPhi = phi[Optimal_ix]
        OptimalRes = ResList[str(Optimal_ix)]
        print("## Deconvolution Finish! ...\n")

        self.info_parameters["phi"] = OptimalPhi
        self.algorithm_matrix = {"B": B, "Xinput_norm": Xinput_norm, "Res": OptimalRes}

        return self

    def predict(self):
        """Prediction function.

        Returns
        -------
        numpy.ndarray
            Predictions of cell-type proportions.

        """
        optim_res = self.algorithm_matrix["Res"]
        prop_pred = optim_res["V"] / optim_res["V"].sum(axis=1, keepdims=True)
        return prop_pred

    def fit_and_predict(self, x, spatial, **kwargs):
        self.fit(x, spatial, **kwargs)
        return self.predict()

    @staticmethod
    def score(x, y):
        """Model performance score measured by mean square error.

        Parameters
        ----------
        x : np.ndarray
            Predicted cell-type proportion matrix (# spots x # cell-types)
        y : np.ndarray
            True cell-type proportion matrix (# spots x # cell-types)

        Returns
        -------
        float
            Mean square error.

        """
        mse = ((x - y / y.sum(1, keepdims=True))**2).mean()
        return mse
