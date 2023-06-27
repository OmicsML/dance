import anndata as ad
import numpy as np
import torch
from networkx.algorithms import bipartite
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.metrics.cluster import normalized_mutual_info_score

from dance import logger
from dance.registers import METRIC_FUNCS, register_metric_func
from dance.typing import Any, Mapping, Optional, Union
from dance.utils.wrappers import torch_to_numpy


def resolve_score_func(score_func: Optional[Union[str, Mapping[Any, float]]]) -> Mapping[Any, float]:
    logger.debug(f"Resolving scoring function from {score_func!r}")
    if score_func is None:
        raise ValueError(f"Scoring function not specified: {score_func=!r}")
    elif isinstance(score_func, str):
        if score_func not in METRIC_FUNCS:
            raise KeyError(f"Failed to obtain scoring function {score_func!r} from the METRI_FUNCS dict, "
                           f"available options are {sorted(METRIC_FUNCS)}")
        score_func = METRIC_FUNCS[score_func]
        logger.debug(f"Scoring function {score_func!r} obtained from the METRIC_FUNCS")
    else:
        logger.debug(f"Input {score_func!r} is not string type, assuming it is a valid score function and return")
    return score_func


@register_metric_func("acc")
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


@register_metric_func("ari")
@torch_to_numpy
def ari(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Adjusted rand index score.

    See
    :func: `sklearn.metrics.adjusted_rand_score`.

    """
    return adjusted_rand_score(true, pred)


@register_metric_func("mse")
@torch_to_numpy
def mse(true: Union[torch.Tensor, np.ndarray], pred: Union[torch.Tensor, np.ndarray]) -> float:
    """Mean squared error score.

    See
    :func: `sklearn.metrics.mean_squared_error`

    """
    return mean_squared_error(true, pred)


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
    matches = bipartite.matching.minimum_weight_full_matching(graph, top_nodes=u)
    best_matches = np.array([matches[x] - len(u) for x in u])
    bipartite_matching_adjacency = np.zeros(raw_logits.shape)
    bipartite_matching_adjacency[np.arange(raw_logits.shape[0]), best_matches] = 1
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
    return NMI_score, ARI_score
