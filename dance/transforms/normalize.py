import os
from multiprocessing import Manager, Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.discrete.discrete_model
import statsmodels.nonparametric.kernel_regression
from KDEpy import FFTKDE
from scipy import stats

from dance.data.base import Data
from dance.transforms.base import BaseTransform
from dance.typing import Dict, List, Literal, NormMode, Optional, Union
from dance.utils.matrix import normalize


class ScaleFeature(BaseTransform):
    """Scale the feature matrix in the AnnData object.

    This is an extension of :meth:`scanpy.pp.scale`, allowing split- or batch-wide scaling.

    Parameters
    ----------
    axis
        Axis along which the scaling is performed.
    split_names
        Indicate which splits to perform the scaling independently. If set to 'ALL', then go through all splits
        available in the data.
    batch_key
        Indicate which column in ``.obs`` to use as the batch index to guide the batch-wide scaling.
    mode
        Scaling mode, see :meth:`dance.utils.matrix.normalize` for more information.
    eps
        Correction fact, see :meth:`dance.utils.matrix.normalize` for more information.

    Note
    ----
    The order of checking split- or batch-wide scaling mode is: batch_key > split_names > None (i.e., all).

    """

    _DISPLAY_ATTRS = ("axis", "mode", "eps", "split_names", "batch_key")

    def __init__(
        self,
        *,
        axis: int = 0,
        split_names: Optional[Union[Literal["ALL"], List[str]]] = None,
        batch_key: Optional[str] = None,
        mode: NormMode = "normalize",
        eps: float = -1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axis = axis
        self.split_names = split_names
        self.batch_key = batch_key
        self.mode = mode
        self.eps = eps

    def _get_idx_dict(self, data) -> List[Dict[str, List[int]]]:
        batch_key = self.batch_key
        split_names = self.split_names

        if batch_key is not None:
            if split_names is not None:
                raise ValueError("Exactly one of split_names and batch_key can be specified, got: "
                                 f"split_names={split_names!r}, batch_key={batch_key!r}")
            elif batch_key not in (avail_opts := data.data.obs.columns.tolist()):
                raise KeyError(f"{batch_key=!r} not found in `.obs`. Available columns are: {avail_opts}")
            batch_col = data.data.obs[batch_key]
            idx_dict = {f"batch:{i}": np.where(batch_col[0] == i)[0].tolist() for i in batch_col[0].unique()}
            return idx_dict

        if split_names is None:
            idx_dict = {"full": list(range(data.shape[0]))}
        elif isinstance(split_names, str) and split_names == "ALL":
            idx_dict = {f"split:{i}": j for i, j in data._split_idx_dict.items()}
        elif isinstance(split_names, list):
            idx_dict = {f"split:{i}": data.get_split_idx(i) for i in split_names}
        else:
            raise TypeError(f"Unsupported type {type(split_names)} for split_names: {split_names!r}")

        return idx_dict

    def __call__(self, data):
        if isinstance(data.data.X, sp.spmatrix):
            self.logger.warning("Native support for sparse matrix is not implemented yet, "
                                "converting to dense array explicitly.")
            data.data.X = data.data.X.A

        idx_dict = self._get_idx_dict(data)
        for name, idx in idx_dict.items():
            self.logger.info(f"Scaling {name} (n={len(idx):,})")
            data.data.X[idx] = normalize(data.data.X[idx], mode=self.mode, axis=self.axis, eps=self.eps)


class ScTransformR(BaseTransform):
    """ScTransform normalization and variance stabiliation.

    Note
    ----
    This is a wrapper for the original R implementation.

    Parameters
    ----------
    min_cells
        Minimum number of cells the gene has to express in, below which that gene will be discarded.

    Reference
    ---------
    https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1874-1

    """

    def __init__(self, min_cells: int = 5, **kwargs):
        self.min_cells = min_cells
        super().__init__(**kwargs)

    def __call__(self, data: Data) -> Data:
        import rpy2.robjects as robjects
        from anndata2ri import py2rpy
        from rpy2.robjects import numpy2ri, pandas2ri, r
        from rpy2.robjects.conversion import localconverter

        if isinstance(data.data.X, sp.spmatrix):
            self.logger.warning("Native support for sparse matrix is not implemented yet, "
                                "converting to dense array explicitly.")
            data.data.X = data.data.X.A
        adata = data.X.copy()

        with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
            sce = py2rpy(adata)
        robjects.r.assign("sce", sce)
        r_code = f'''
        counts <- assay(sce, "X")
        libsizes <- colSums(counts)
        size.factors <- libsizes/mean(libsizes)
        logcounts(sce) <- log2(t(t(counts)/size.factors) + 1)
        seurat <- as.Seurat(sce,counts="X")
        seurat@assays$RNA<-seurat@assays$originalexp
        seurat_p=SCTransform(seurat, vst.flavor = "v2", verbose = FALSE,min_cells={self.min_cells})
        '''
        r(r_code)
        r_floatmatrix = r('seurat@assays$RNA@data')
        with localconverter(robjects.default_converter + pandas2ri.converter + numpy2ri.converter):
            floatmatrix_df = pandas2ri.rpy2py(r_floatmatrix)

        # Convert to anndata
        adata.X = floatmatrix_df.T
        data.data = adata
        return data


class ScTransform(BaseTransform):
    """ScTransform normalization and variance stabiliation.

    Note
    ----
    This is a Python implementation adapted from https://github.com/atarashansky/SCTransformPy

    Parameters
    ----------
    split_names
        Which split(s) to apply the transformation.
    batch_key
        Key for batch information.
    min_cells
        Minimum number of cells the gene has to express in, below which that gene will be discarded.
    gmean_eps
        Pseudocount.
    n_genes
        Maximum number of genes to use. Use all if set to ``None``.
    n_cells
        maximum number of cells to use. Use all if set to ``None``.
    bin_size
        Number of genes a single bin contain.
    bw_adjust
        Bandwidth adjusting parameter.

    Reference
    ---------
    https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1874-1

    """

    _DISPLAY_ATTRS = ("mode", "eps", "split_names", "batch_key")

    def __init__(
        self,
        split_names: Optional[Union[Literal["ALL"], List[str]]] = None,
        batch_key: Optional[str] = None,
        min_cells: int = 5,
        gmean_eps: int = 1,
        n_genes: Optional[int] = 2000,
        n_cells: Optional[int] = None,
        bin_size: int = 500,
        bw_adjust: float = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.split_names = split_names
        self.batch_key = batch_key
        self.min_cells = min_cells
        self.gmean_eps = gmean_eps
        self.n_genes = n_genes
        self.n_cells = n_cells
        self.bin_size = bin_size
        self.bw_adjust = bw_adjust

    def _get_idx_dict(self, data) -> List[Dict[str, List[int]]]:
        # TODO: refactor out this function; reduce ropied code.
        batch_key = self.batch_key
        split_names = self.split_names

        if batch_key is not None:
            if split_names is not None:
                raise ValueError("Exactly one of split_names and batch_key can be specified, got: "
                                 f"split_names={split_names!r}, batch_key={batch_key!r}")
            elif batch_key not in (avail_opts := data.data.obs.columns.tolist()):
                raise KeyError(f"{batch_key=!r} not found in `.obs`. Available columns are: {avail_opts}")
            batch_col = data.data.obs[batch_key]
            idx_dict = {f"batch:{i}": np.where(batch_col[0] == i)[0].tolist() for i in batch_col[0].unique()}
            return idx_dict

        if split_names is None:
            idx_dict = {"full": list(range(data.shape[0]))}
        elif isinstance(split_names, str) and split_names == "ALL":
            idx_dict = {f"split:{i}": j for i, j in data._split_idx_dict.items()}
        elif isinstance(split_names, list):
            idx_dict = {f"split:{i}": data.get_split_idx(i) for i in split_names}
        else:
            raise TypeError(f"Unsupported type {type(split_names)} for split_names: {split_names!r}")

        return idx_dict

    def __call__(self, data):
        if isinstance(data.data.X, sp.spmatrix):
            self.logger.warning("Native support for sparse matrix is not implemented yet, "
                                "converting to dense array explicitly.")
            data.data.X = data.data.X.A

        # idx_dict = self._get_idx_dict(data)
        # for name, idx in idx_dict.items():
        selected_data = data.data.copy()
        X = selected_data.X.copy()
        X = sp.csr_matrix(X)
        X.eliminate_zeros()
        gn = np.array(list(selected_data.var_names))
        cn = np.array(list(selected_data.obs_names))
        genes_cell_count = X.sum(0).A.flatten()
        genes = np.where(genes_cell_count >= self.min_cells)[0]
        genes_ix = genes.copy()
        X = X[:, genes]
        gn = gn[genes]
        genes = np.arange(X.shape[1])
        genes_cell_count = X.sum(0).A.flatten()
        genes_log_gmean = np.log10(gmean(X, axis=0, eps=self.gmean_eps))

        if self.n_cells is not None and self.n_cells < X.shape[0]:
            cells_step1 = np.sort(np.random.choice(X.shape[0], replace=False, size=self.n_cells))
            genes_cell_count_step1 = X[cells_step1].sum(0).A.flatten()
            genes_step1 = np.where(genes_cell_count_step1 >= self.min_cells)[0]
            genes_log_gmean_step1 = np.log10(gmean(X[cells_step1][:, genes_step1], axis=0, eps=self.gmean_eps))
        else:
            cells_step1 = np.arange(X.shape[0])
            genes_step1 = genes
            genes_log_gmean_step1 = genes_log_gmean

        umi = X.sum(1).A.flatten()
        log_umi = np.log10(umi)
        X2 = X.copy()
        X2.data[:] = 1
        gene = X2.sum(1).A.flatten()
        log_gene = np.log10(gene)
        umi_per_gene = umi / gene
        log_umi_per_gene = np.log10(umi_per_gene)

        cell_attrs = pd.DataFrame(
            index=cn, data=np.vstack((umi, log_umi, gene, log_gene, umi_per_gene, log_umi_per_gene)).T,
            columns=['umi', 'log_umi', 'gene', 'log_gene', 'umi_per_gene', 'log_umi_per_gene'])  # yapf: disable

        data_step1 = cell_attrs.iloc[cells_step1]

        if self.n_genes is not None and self.n_genes < len(genes_step1):
            log_gmean_dens = stats.gaussian_kde(genes_log_gmean_step1, bw_method='scott')
            xlo = np.linspace(genes_log_gmean_step1.min(), genes_log_gmean_step1.max(), 512)
            ylo = log_gmean_dens.evaluate(xlo)
            xolo = genes_log_gmean_step1
            sampling_prob = 1 / (np.interp(xolo, xlo, ylo) + _EPS)
            genes_step1 = np.sort(
                np.random.choice(genes_step1, size=self.n_genes, p=sampling_prob / sampling_prob.sum(), replace=False))
            genes_log_gmean_step1 = np.log10(gmean(X[cells_step1, :][:, genes_step1], eps=self.gmean_eps))

        bin_ind = np.ceil(np.arange(1, genes_step1.size + 1) / self.bin_size)
        max_bin = max(bin_ind)

        ps = Manager().dict()

        for i in range(1, int(max_bin) + 1):
            genes_bin_regress = genes_step1[bin_ind == i]
            umi_bin = X[cells_step1, :][:, genes_bin_regress]

            mm = np.vstack((np.ones(data_step1.shape[0]), data_step1['log_umi'].values.flatten())).T

            pc_chunksize = umi_bin.shape[1] // os.cpu_count() + 1
            pool = Pool(os.cpu_count(), _parallel_init, [genes_bin_regress, umi_bin, gn, mm, ps])
            try:
                pool.map(_parallel_wrapper, range(umi_bin.shape[1]), chunksize=pc_chunksize)
            finally:
                pool.close()
                pool.join()

        ps = ps._getvalue()

        model_pars = pd.DataFrame(data=np.vstack([ps[x] for x in gn[genes_step1]]),
                                  columns=['Intercept', 'log_umi', 'theta'], index=gn[genes_step1])
        min_theta = 1e-7
        x = model_pars['theta'].values.copy()
        x[x < min_theta] = min_theta
        model_pars['theta'] = x
        dispersion_par = np.log10(1 + 10**genes_log_gmean_step1 / model_pars['theta'].values.flatten())

        model_pars = model_pars.iloc[:, model_pars.columns != 'theta'].copy()
        model_pars['dispersion'] = dispersion_par

        outliers = np.vstack(
            [is_outlier(model_pars.values[:, i], genes_log_gmean_step1) for i in range(model_pars.shape[1])]).sum(0) > 0

        filt = np.invert(outliers)
        model_pars = model_pars[filt]
        genes_step1 = genes_step1[filt]
        genes_log_gmean_step1 = genes_log_gmean_step1[filt]

        z = FFTKDE(kernel='gaussian', bw='ISJ').fit(genes_log_gmean_step1)
        z.evaluate()
        bw = z.bw * self.bw_adjust
        x_points = np.vstack((genes_log_gmean, np.array([min(genes_log_gmean_step1)] * genes_log_gmean.size))).max(0)
        x_points = np.vstack((x_points, np.array([max(genes_log_gmean_step1)] * genes_log_gmean.size))).min(0)
        full_model_pars = pd.DataFrame(data=np.zeros((x_points.size, model_pars.shape[1])), index=gn,
                                       columns=model_pars.columns)

        for i in model_pars.columns:
            kr = statsmodels.nonparametric.kernel_regression.KernelReg(model_pars[i].values,
                                                                       genes_log_gmean_step1[:, None], ['c'],
                                                                       reg_type='ll', bw=[bw])
            full_model_pars[i] = kr.fit(data_predict=x_points)[0]
        theta = 10**genes_log_gmean / (10**full_model_pars['dispersion'].values - 1)
        full_model_pars['theta'] = theta
        del full_model_pars['dispersion']
        d = X.data
        x, y = X.nonzero()
        mud = np.exp(full_model_pars.values[:, 0][y] +
                     full_model_pars.values[:, 1][y] * cell_attrs['log_umi'].values[x])
        vard = mud + mud**2 / full_model_pars['theta'].values.flatten()[y]
        X.data[:] = (d - mud) / vard**0.5
        X.data[X.data < 0] = 0
        X.eliminate_zeros()
        clip = np.sqrt(X.shape[0] / 30)
        X.data[X.data > clip] = clip
        selected_data.raw = selected_data.copy()
        d = dict(zip(np.arange(X.shape[1]), genes_ix))
        x, y = X.nonzero()
        y = np.array([d[i] for i in y])
        data = X.data
        Xnew = sp.coo_matrix((data, (x, y)), shape=selected_data.shape).tocsr()
        selected_data.X = Xnew
        for c in full_model_pars.columns:
            selected_data.var[c + '_sct'] = full_model_pars[c]

        for c in cell_attrs.columns:
            selected_data.obs[c + '_sct'] = cell_attrs[c]

        for c in model_pars.columns:
            selected_data.var[c + '_step1_sct'] = model_pars[c]
        print(selected_data)

        z = pd.Series(index=gn, data=np.zeros(gn.size, dtype='int'))
        z[gn[genes_step1]] = 1

        w = pd.Series(index=gn, data=np.zeros(gn.size, dtype='int'))
        w[gn] = genes_log_gmean
        selected_data.var['genes_step1_sct'] = z
        selected_data.var['log10_gmean_sct'] = w

        return selected_data


def gmean(X, axis=0, eps=1):
    X = X.copy()
    X.data[:] = np.log(X.data + eps)
    return np.exp(X.mean(axis).A.flatten()) - eps


_EPS = np.finfo(float).eps


def robust_scale_binned(y, x, breaks):
    bins = np.digitize(x, breaks)
    binsu = np.unique(bins)
    res = np.zeros(bins.size)
    for i in range(binsu.size):
        yb = y[bins == binsu[i]]
        res[bins == binsu[i]] = (yb - np.median(yb)) / (1.4826 * np.median(np.abs(yb - np.median(yb))) + _EPS)

    return res


def is_outlier(y, x, th=10):
    z = FFTKDE(kernel='gaussian', bw='ISJ').fit(x)
    z.evaluate()
    bin_width = (max(x) - min(x)) * z.bw / 2
    eps = _EPS * 10

    breaks1 = np.arange(min(x), max(x) + bin_width, bin_width)
    breaks2 = np.arange(min(x) - eps - bin_width / 2, max(x) + bin_width, bin_width)
    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)
    return np.abs(np.vstack((score1, score2))).min(0) > th


def _parallel_init(igenes_bin_regress, iumi_bin, ign, imm, ips):
    global genes_bin_regress
    global umi_bin
    global gn
    global mm
    global ps
    genes_bin_regress = igenes_bin_regress
    umi_bin = iumi_bin
    gn = ign
    mm = imm
    ps = ips


def _parallel_wrapper(j):
    name = gn[genes_bin_regress[j]]
    y = umi_bin[:, j].A.flatten()
    pr = statsmodels.discrete.discrete_model.Poisson(y, mm)
    res = pr.fit(disp=False)
    mu = res.predict()
    theta = theta_ml(y, mu)
    ps[name] = np.append(res.params, theta)


def theta_ml(y, mu):
    n = y.size
    weights = np.ones(n)
    limit = 10
    eps = (_EPS)**0.25

    from scipy.special import polygamma, psi

    def score(n, th, mu, y, w):
        return sum(w * (psi(th + y) - psi(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th)))

    def info(n, th, mu, y, w):
        return sum(w * (-polygamma(1, th + y) + polygamma(1, th) - 1 / th + 2 / (mu + th) - (y + th) / (mu + th)**2))

    t0 = n / sum(weights * (y / mu - 1)**2)
    it = 0
    de = 1

    while (it + 1 < limit and abs(de) > eps):
        it += 1
        t0 = abs(t0)
        i = info(n, t0, mu, y, weights)
        de = score(n, t0, mu, y, weights) / i
        t0 += de
    t0 = max(t0, 0)

    return t0
