from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
import torch
from networkx.algorithms import bipartite
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.metrics.cluster import normalized_mutual_info_score

from dance import logger
from dance.registry import REGISTERED_METRIC_FUNCS, register_metric_func
from dance.typing import Any, Mapping, Optional, Union
from dance.utils.wrappers import torch_to_numpy


def resolve_score_func(score_func: Optional[Union[str, Mapping[Any, float]]]) -> Mapping[Any, float]:
    logger.debug(f"Resolving scoring function from {score_func!r}")
    if score_func is None:
        raise ValueError(f"Scoring function not specified: {score_func=!r}")
    elif isinstance(score_func, str):
        if score_func not in REGISTERED_METRIC_FUNCS:
            raise KeyError(f"Failed to obtain scoring function {score_func!r} from the METRI_FUNCS dict, "
                           f"available options are {sorted(REGISTERED_METRIC_FUNCS)}")
        score_func = REGISTERED_METRIC_FUNCS[score_func]
        logger.debug(f"Scoring function {score_func!r} obtained from REGISTERED_METRIC_FUNCS")
    else:
        logger.debug(f"Input {score_func!r} is not string type, assuming it is a valid score function and return")
    return score_func


@register_metric_func()
@torch_to_numpy
def acc(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Accuracy score.

    This specific implementation of accuracy score accounts for the possibility where the true label for an instance
    may contain multiple positives. This happens because in some cases of cell type annotation tasks, some cells in
    the test set have slightly more ambiguous cell-type annotations than the training set.

    Parameters
    ----------
    pred
        Predicted labels.
    true
        True labels. Can be either a maxtrix of size (samples x labels) with ones indicating positives, or a
        vector of size (sameples x 1) where each element is the index of the corresponding label for the sample.
        The first option provides flexibility to cases where a sample could be associated with multiple labels
        at test time while the model was trained as a multi-class classifier.

    Returns
    -------
    float
        Accuracy score.

    """
    return true[np.arange(pred.shape[0]), pred.ravel()].mean()


@register_metric_func()
@torch_to_numpy
def ari(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Adjusted rand index score.

    See
    :func: `sklearn.metrics.adjusted_rand_score`.

    """
    return adjusted_rand_score(true, pred)


@register_metric_func()
@torch_to_numpy
def mse(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean squared error score.

    See
    :func: `sklearn.metrics.mean_squared_error`

    """
    return mean_squared_error(true, pred)


@register_metric_func()
@torch_to_numpy
def mape(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean absolute percentage error score.

    See
    :func: `sklearn.metrics.mean_absolute_percentage_error`

    """
    return mean_absolute_percentage_error(true, pred)


def get_bipartite_matching_adjacency_matrix_mk3(raw_logits, threshold_quantile=0.995, copy=False):
    #getting rid of unpromising graph connections
    if copy:
        weights = raw_logits.copy()
    else:
        weights = raw_logits
    quantile_row = np.quantile(weights, threshold_quantile, axis=0, keepdims=True)
    quantile_col = np.quantile(weights, threshold_quantile, axis=1, keepdims=True)
    #quantile_minimum = np.minimum(quantile_row, quantile_col, out=quantile_row)
    mask_ = (weights < quantile_row)
    mask_ = np.logical_and(mask_, (weights < quantile_col), out=mask_)
    #weights[weights<quantile_minimum] = 0
    weights[mask_] = 0
    weights_sparse = sparse.csr_matrix(-weights)
    del (weights)
    graph = bipartite.matrix.from_biadjacency_matrix(weights_sparse)
    #explicitly combining top nodes in once component or networkx freaks tf out
    u = [n for n in graph.nodes if graph.nodes[n]['bipartite'] == 0]
    try:
        matches = bipartite.matching.minimum_weight_full_matching(graph, top_nodes=u)
        best_matches = np.array([matches[x] - len(u) for x in u])
        bipartite_matching_adjacency = np.zeros(raw_logits.shape)
        bipartite_matching_adjacency[np.arange(raw_logits.shape[0]), best_matches] = 1
    except:
        # bipartite_matching_adjacency = weights_sparse / np.sum(1, keepdims=True)
        bipartite_matching_adjacency = weights_sparse.toarray()
    return bipartite_matching_adjacency


def batch_separated_bipartite_matching(batch1, batch2, emb1, emb2, threshold_quantile):
    matrix = np.zeros((batch1.shape[0], batch2.shape[0]))
    for b in batch1.unique():
        i0 = (batch1 == b).values.nonzero()[0].tolist()
        j0 = (batch2 == b).values.nonzero()[0].tolist()
        logits = torch.matmul(emb1[i0], emb2[j0].T)
        logits = torch.softmax(logits, -1) + torch.softmax(logits, 0)
        logits = logits.cpu().numpy()

        out1_2 = get_bipartite_matching_adjacency_matrix_mk3(logits, threshold_quantile=threshold_quantile)
        matrix[np.ix_(i0, j0)] = out1_2
    return matrix


def labeled_clustering_evaluate(adata: ad.AnnData, test_sol: ad.AnnData, cluster: int = 10):
    kmeans = KMeans(n_clusters=cluster, n_init=5, random_state=200)

    true_labels = test_sol.obs['cell_type'].to_numpy()
    pred_labels = kmeans.fit_predict(adata.X)
    NMI_score = round(normalized_mutual_info_score(true_labels, pred_labels, average_method='max'), 3)
    ARI_score = round(adjusted_rand_score(true_labels, pred_labels), 3)

    print('NMI: ' + str(NMI_score) + ' ARI: ' + str(ARI_score))
    return {'dance_nmi': NMI_score, 'dance_ari': ARI_score}


# Reference: https://github.com/openproblems-bio/neurips2021_multimodal_viash/tree/main/src/joint_embedding/metrics
def integration_openproblems_evaluate(adata: ad.AnnData):
    import scanpy as sc
    import scib
    score = {}
    if 'X_emb' not in adata.obsm:
        adata.obsm['X_emb'] = adata.X
    score['asw_batch'] = scib.me.silhouette_batch(adata, batch_key='batch', group_key='cell_type', embed='X_emb',
                                                  verbose=False)
    score['asw_label'] = scib.me.silhouette(adata, group_key='cell_type', embed='X_emb')
    # nmi
    sc.pp.neighbors(adata, use_rep='X_emb')
    scib.cl.opt_louvain(adata, label_key='cell_type', cluster_key='cluster', plot=False, inplace=True, force=True)
    score['nmi'] = scib.me.nmi(adata, group1='cluster', group2='cell_type')
    # cc_cons
    recompute_cc = 'S_score' not in adata.obs_keys() or \
                    'G2M_score' not in adata.obs_keys()
    score['cc_cons'] = scib.me.cell_cycle(adata_pre=adata, adata_post=adata, batch_key='batch', embed='X_emb',
                                          recompute_cc=recompute_cc, organism=adata.uns['organism'])
    # ti_cons
    obs_keys = adata.obs_keys()
    adt_atac_trajectory = 'pseudotime_order_ATAC' if 'pseudotime_order_ATAC' in obs_keys else 'pseudotime_order_ADT'
    if 'pseudotime_order_GEX' in obs_keys:
        score_rna = scib.me.trajectory_conservation(adata_pre=adata, adata_post=adata, label_key='cell_type',
                                                    pseudotime_key='pseudotime_order_GEX')
        score_rna_batch = scib.me.trajectory_conservation(adata_pre=adata, adata_post=adata, label_key='cell_type',
                                                          batch_key='batch', pseudotime_key='pseudotime_order_GEX')
    else:
        score_rna = score_rna_batch = np.nan
    if adt_atac_trajectory in obs_keys:
        score_adt_atac = scib.me.trajectory_conservation(adata_pre=adata, adata_post=adata, label_key='cell_type',
                                                         pseudotime_key=adt_atac_trajectory)
        score_adt_atac_batch = scib.me.trajectory_conservation(adata_pre=adata, adata_post=adata, label_key='cell_type',
                                                               batch_key='batch', pseudotime_key=adt_atac_trajectory)
    else:
        score_adt_atac = score_adt_atac_batch = np.nan
    #score['ti_cons_mean'] = (score_rna + score_adt_atac) / 2
    score['ti_cons_batch_mean'] = (score_rna_batch + score_adt_atac) / 2
    score['graph_conn'] = scib.me.graph_connectivity(adata, label_key='cell_type')
    score['final_scores'] = sum(score.values()) / len(score)
    return score


@register_metric_func()
@torch_to_numpy
def silhouette(X: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Silhouette Coefficient.

    A higher Silhouette Coefficient score relates to a model with better defined clusters.
    The score is higher when clusters are dense and well separated.

    See
    :func: `sklearn.metrics.silhouette_score`.

    """
    # 轮廓系数要求簇的数量至少为2，最多为 n_samples - 1
    n_labels = len(np.unique(labels))
    n_samples = len(X)
    if not (2 <= n_labels < n_samples):
        # 如果簇的数量不满足要求（例如所有点都分到一类），返回一个无意义的差值
        # 返回0或-1都是常见做法，这里返回0表示中性
        print(f"Warning: Silhouette score is not defined for n_labels={n_labels}. Returning 0.")
        return 0.0

    return silhouette_score(X, labels)


@register_metric_func()
@torch_to_numpy
def calinski_harabasz(X: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Calinski-Harabasz Index (Variance Ratio Criterion).

    The score is higher when clusters are dense and well separated. It is computed
    as the ratio of the sum of between-cluster dispersion and of within-cluster dispersion.

    See
    :func: `sklearn.metrics.calinski_harabasz_score`.

    """
    # 与轮廓系数类似，CH指数也要求簇的数量在[2, n_samples-1]之间
    n_labels = len(np.unique(labels))
    n_samples = len(X)
    if not (2 <= n_labels < n_samples):
        print(f"Warning: Calinski-Harabasz score is not defined for n_labels={n_labels}. Returning 0.")
        return 0.0

    return calinski_harabasz_score(X, labels)


@register_metric_func()
@torch_to_numpy
def davies_bouldin(X: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]) -> float:
    """Davies-Bouldin Index.

    The score is defined as the average similarity measure of each cluster with its
    most similar cluster, where similarity is the ratio of within-cluster distances
    to between-cluster distances. Thus, clusters which are farther apart and less
    dispersed will result in a better score. A lower value is better.

    See
    :func: `sklearn.metrics.davies_bouldin_score`.

    """
    # DB指数同样要求簇的数量在[2, n_samples-1]之间
    n_labels = len(np.unique(labels))
    n_samples = len(X)
    if not (2 <= n_labels < n_samples):
        print(f"Warning: Davies-Bouldin score is not defined for n_labels={n_labels}. Returning 0.")
        return 0.0

    return davies_bouldin_score(X, labels)


METRIC_BOUNDS = {
    'silhouette': {
        'min': -1.0,
        'max': 1.0
    },
    'calinski_harabasz': {
        'min': 0.0,
        'max': 20000.0
    },
    'davies_bouldin': {
        'min': 0.0,
        'max': 10.0
    }
}


def calculate_unified_scores(score_dict: Dict[str, float]) -> float:
    """Evaluates performance with logging for out-of-bounds values before clipping."""
    # --- 1. Silhouette Score ---
    s_val = score_dict['silhouette']
    s_bounds = METRIC_BOUNDS['silhouette']
    if not (s_bounds['min'] <= s_val <= s_bounds['max']):
        logger.info(f"Silhouette score {s_val:.4f} is outside the defined bounds "
                    f"[{s_bounds['min']}, {s_bounds['max']}]. Clipping.")
    s_val_clipped = np.clip(s_val, s_bounds['min'], s_bounds['max'])
    norm_s = (s_val_clipped - s_bounds['min']) / (s_bounds['max'] - s_bounds['min'])

    # --- 2. Calinski-Harabasz Score ---
    ch_val = score_dict['calinski_harabasz']
    ch_bounds = METRIC_BOUNDS['calinski_harabasz']
    if not (ch_bounds['min'] <= ch_val <= ch_bounds['max']):
        logger.info(f"Calinski-Harabasz score {ch_val:.2f} is outside the defined bounds "
                    f"[{ch_bounds['min']}, {ch_bounds['max']}]. Clipping.")
    ch_val_clipped = np.clip(ch_val, ch_bounds['min'], ch_bounds['max'])
    norm_ch = (ch_val_clipped - ch_bounds['min']) / (ch_bounds['max'] - ch_bounds['min'])

    # --- 3. Davies-Bouldin Score ---
    db_val = score_dict['davies_bouldin']
    db_bounds = METRIC_BOUNDS['davies_bouldin']
    if not (db_bounds['min'] <= db_val <= db_bounds['max']):
        logger.info(f"Davies-Bouldin score {db_val:.4f} is outside the defined bounds "
                    f"[{db_bounds['min']}, {db_bounds['max']}]. Clipping.")
    db_val_clipped = np.clip(db_val, db_bounds['min'], db_bounds['max'])
    norm_db = (db_bounds['max'] - db_val_clipped) / (db_bounds['max'] - db_bounds['min'])

    unified_score = (norm_s + norm_ch + norm_db) / 3.0
    return unified_score
