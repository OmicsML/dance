import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import r2_score

from dance import logger
from dance.transforms.base import BaseTransform
from dance.transforms.stats import genestats_alpha, genestats_mu
from dance.typing import Dict, List, Optional, Tuple


class SCNFeature(BaseTransform):
    """Differential gene-pair feature used in SingleCellNet."""

    _DISPLAY_ATTRS = ("num_top_genes", "alpha1", "alpha2", "mu", "num_top_gene_pairs", "max_gene_per_ct", "split_name")

    def __init__(self, num_top_genes: int = 10, alpha1: float = 0.05, alpha2: float = 0.001, mu: float = 2,
                 num_top_gene_pairs: int = 25, max_gene_per_ct: int = 3, *, split_name: Optional[str] = "train",
                 channel: Optional[str] = None, channel_type: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.num_top_genes = num_top_genes
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.mu = mu

        self.num_top_gene_pairs = num_top_gene_pairs
        self.max_gene_per_ct = max_gene_per_ct

        self.split_name = split_name
        self.channel = channel
        self.channel_type = channel_type

    def __call__(self, data):
        split_idx = data.get_split_idx(self.split_name)
        all_exp_df = data.data.to_df(self.channel)  # TODO: return as numpy or sparse csr to improve efficiency?
        cell_type_df = data.get_feature(return_type="default", channel="cell_type", channel_type="obsm").iloc[split_idx]

        # Get normalized features
        adata = data.data[split_idx].copy()
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.highly_variable_genes(adata, max_mean=4, subset=True)
        sc.pp.scale(adata, max_value=10)
        norm_exp_df = adata.to_df()
        cell_type_df = cell_type_df.loc[adata.obs_names]  # not necessary, but kept here in case we subsample cells

        # Get differentially expressed genes and gene pairs
        cell_type_array = cell_type_df.columns.values[cell_type_df.values.argmax(1)]
        degs_dict = get_diff_exp_genes(norm_exp_df, cell_type_array, alpha1=self.alpha1, alpha2=self.alpha2, mu=self.mu,
                                       num_top_genes=self.num_top_genes)
        top_gene_pairs = get_top_gene_pairs(norm_exp_df, cell_type_array, degs_dict,
                                            num_top_pairs=self.num_top_gene_pairs, max_gene_per_ct=self.max_gene_per_ct)

        # Prepare binarized feature using the selected gene pairs
        scn_feat = query_transform(all_exp_df, top_gene_pairs)
        data.data.obsm[self.out] = scn_feat

        return data


# # TODO: move to dance.transforms.tools? or make it a transform?
# def binGenes(geneStats, nbins=20, meanType="overall_mean"):
#     max = np.max(geneStats[meanType])
#     min = np.min(geneStats[meanType])
#     rrange = max - min
#     inc = rrange / nbins
#     threshs = np.arange(max, min, -1 * inc)
#     res = pd.DataFrame(index=geneStats.index.values, data=np.arange(0, geneStats.index.size, 1), columns=["bin"])
#     for i in range(0, len(threshs)):
#         res.loc[geneStats[meanType] <= threshs[i], "bin"] = len(threshs) - i
#     return res


def query_transform(exp_df: pd.DataFrame, gene_pairs: List[Tuple[str, str]]):
    """Transform expression data into SCN feature given selected gene pairs.

    Parameters
    ----------
    exp_df
        Expression matrix (sample x gene).
    gene_pairs
        List of selected top differentiating gene pairs.

    Returns
    -------
    gene_pair_diff_bin
        SCN feature. A binary matrix indicating whether the source genes have higher expression than the target genes
        in the top selected gene pairs.

    """
    genes1, genes2 = map(list, zip(*gene_pairs))
    gene_pair_diff_bin = (exp_df[genes1].values > exp_df[genes2].values).astype(float)
    gene_pair_diff_bin = pd.DataFrame(gene_pair_diff_bin, index=exp_df.index, columns=map("&".join, gene_pairs))
    return gene_pair_diff_bin


def get_top_gene_pairs(exp_df: pd.DataFrame, cell_type_array: np.ndarray, degs_dict: Dict[str, List[str]], *,
                       num_top_pairs: int = 250, max_gene_per_ct: int = 3) -> List[Tuple[str, str]]:
    """Obtain top differentiating gene pairs.

    Parameters
    ----------
    exp_df
        Expression matrix (sample x gene).
    cell_type_array
        1-d array of cell-type information for each sample.
    degs_dict
        Dictionary of differentially expressed genes for each cell type.
    num_top_pairs
        Number of top differentiating gene pairs to get.
    max_gene_per_ct
        Maximum number of genes allowed to be attributed to a cell type (in the form of gene pairs).

    Returns
    -------
    top_gene_pairs
        List of top differentiating gene pairs.

    """
    top_gene_pairs = []
    for cell_type, degs in degs_dict.items():
        logger.info(f"Extracting top gene pairs for {cell_type}...")
        logger.debug(f"All DEGs:\n{degs}")
        logger.info(f"\tFirst five DEGs: {', '.join(degs[:5])}")

        gene_pairs = list(itertools.combinations(degs, 2))
        pair_df = pd.DataFrame(gene_pairs, columns=["gene1", "gene2"])
        pair_df["gene_pair"] = pair_df.apply("&".join, axis=1)

        gene_pair_diff_bin = (exp_df[pair_df["gene1"]].values > exp_df[pair_df["gene2"]].values).astype(float)
        gene_pair_diff_bin = pd.DataFrame(gene_pair_diff_bin, columns=pair_df["gene_pair"])

        cell_type_mask = np.zeros(cell_type_array.size)
        cell_type_mask[cell_type_array == cell_type] = 1

        gene_pair_scores = _get_deg_scores(gene_pair_diff_bin, cell_type_mask)
        best_gene_pairs = _get_best_gene_pairs(gene_pair_scores, gene_pairs, num_pairs=num_top_pairs,
                                               max_gene_per_ct=max_gene_per_ct)

        best_gene_pairs_str = ", ".join(["&".join(gene_pair) for gene_pair in best_gene_pairs[:5]])
        logger.info(f"\tFirst five gene pairs: {best_gene_pairs_str}")

        top_gene_pairs.extend(best_gene_pairs)

    top_gene_pairs = sorted(set(top_gene_pairs))

    return top_gene_pairs


def _get_best_gene_pairs(gene_pair_scores: np.ndarray, gene_pairs: List[Tuple[str, str]], num_pairs: int = 50,
                         max_gene_per_ct: int = 3) -> List[Tuple[str, str]]:
    valid_idx = np.where(~np.isnan(gene_pair_scores))[0]
    sorted_idx = valid_idx[gene_pair_scores[valid_idx].argsort()[::-1]]

    best_gene_pairs = []
    count_dict = defaultdict(int)
    for idx in sorted_idx:
        gene_pair = gene1, gene2 = gene_pairs[idx]

        if (count_dict[gene1] < max_gene_per_ct) and (count_dict[gene2] < max_gene_per_ct):
            best_gene_pairs.append(gene_pair)
            count_dict[gene1] += 1
            count_dict[gene2] += 1

        if len(best_gene_pairs) == num_pairs:
            break

    else:  # did not obtain enough number of gene pairs required
        logger.warning(f"Ran out of gene pairs to select (total_pairs={sorted_idx.size:,}), target number: "
                       f"{num_pairs:,}, number of gene pairs selected: {len(best_gene_pairs):,}")

    return best_gene_pairs


def get_diff_exp_genes(exp_df: pd.DataFrame, cell_type_array: np.ndarray, *, num_top_genes: int = 100,
                       threshold: float = 0, alpha1: float = 0.05, alpha2: float = 0.001,
                       mu: float = 2) -> Tuple[Dict[str, List[str]], List[str]]:
    """Get differntially expressed genes via regression.

    Parameters
    ----------
    exp_df
        Expression matrix (sample x gene).
    cell_type_array
        1-d array of cell-type information for each sample.
    num_top_genes
        Number of top differentially expressed genes to use.
    threshold
        Gene expression threshold parameters.
    alpha1
        Alpha 1 threshold parameter.
    alpha2
        Alpha 2 threshold parameter.
    mu
        mu threshold parameter.

    Returns
    -------
    degs_dict
        Dictionary of selected top differentially expressed genes for each cell type.

    """
    alpha_df = genestats_alpha(exp_df, threshold=threshold)
    mu_df = genestats_mu(exp_df, threshold=threshold)

    cond1 = alpha_df > alpha1
    cond2 = alpha_df > alpha2
    cond3 = mu_df > mu
    indicator = np.logical_or(cond1, np.logical_and(cond2, cond3))
    selected_genes = exp_df.columns.values[indicator]

    degs_dict = _get_degs_dict(exp_df.loc[:, selected_genes], cell_type_array, num_top_genes)

    return degs_dict


def _get_degs_dict(exp_df, cell_type_array, num_top_genes, both_ends: bool = True) -> Dict[str, List[str]]:
    # NOTE: when both_ends is set, the actual number of selected genes will be at most doubled
    degs_dict = {}
    for cell_type in np.unique(cell_type_array):
        cell_type_mask = np.zeros(cell_type_array.size)
        cell_type_mask[cell_type_array == cell_type] = 1

        cval = _get_deg_scores(exp_df, cell_type_mask)
        valid_idx = np.where(~np.isnan(cval))[0]
        sorted_idx = cval[valid_idx].argsort()[::-1]

        # Select positively differentially expressed genes
        selected_sorted_idx = sorted_idx[:num_top_genes].tolist()
        if both_ends:  # add negatively differentially expressed genes
            selected_sorted_idx.extend(sorted_idx[-num_top_genes:].tolist())

        selected = valid_idx[sorted(set(selected_sorted_idx))]

        degs_dict[cell_type] = exp_df.columns[selected].tolist()

    return degs_dict


def _get_deg_scores(exp_df, cell_type_mask) -> np.ndarray:
    y = np.vstack([cell_type_mask, np.ones(len(cell_type_mask))]).T
    p = np.linalg.lstsq(y, exp_df, rcond=None)[0]

    exp_recon = y @ p
    r2 = r2_score(exp_df.values, exp_recon, multioutput="raw_values").clip(0)
    cval = np.sqrt(r2) * np.sign(p[0])

    return cval
