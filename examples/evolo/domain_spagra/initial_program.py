import argparse
import time  # 新增：导入 time 模块

import numpy as np
import pandas as pd
import sklearn

from dance.datasets.spatial import SpatialLIBDDataset
from dance.modules.spatial.spatial_domain.spaGRA import SpaGRA
from dance.registry import register_preprocessor
from dance.transforms import Compose, HighlyVariableGenesRawCount, NormalizeTotalLog1P, PrefilterGenes, SetConfig
from dance.transforms.base import BaseTransform
from dance.typing import LogLevel, Optional
from dance.utils import set_seed, sub_data
from dance.utils.metrics import calculate_unified_scores, resolve_score_func


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "spatial", overwrite=True)
class CalSpatialNet(BaseTransform):
    """Construct the spatial neighbor networks.

    Parameters
    ----------
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    spatial_uns
        Key for storing spatial networks in adata.uns. Default is "Spatial_Net".

    """

    _DISPLAY_ATTRS = ("rad_cutoff", "k_cutoff", "model", "spatial_uns")

    def __init__(self, rad_cutoff=None, k_cutoff=None, model='Radius', spatial_uns="Spatial_Net", out=None,
                 log_level="WARNING"):
        super().__init__(out=out, log_level=log_level)
        self.rad_cutoff = rad_cutoff
        self.k_cutoff = k_cutoff
        self.model = model
        self.spatial_uns = spatial_uns

    def __call__(self, data):
        """Construct spatial neighbor networks."""
        adata = data.data

        print('------Calculating spatial graph...')
        assert (self.model in ['Radius', 'KNN'])
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']

        if self.model == 'Radius':
            nbrs = sklearn.neighbors.NearestNeighbors(radius=self.rad_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

        if self.model == 'KNN':
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=self.k_cutoff + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[
            Spatial_Net['Distance'] > 0,
        ]
        id_cell_trans = dict(zip(
            range(coor.shape[0]),
            np.array(coor.index),
        ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

        adata.uns[self.spatial_uns] = Spatial_Net
        return data


# EVOLVE-BLOCK-END


def get_preprocessing_pipeline(log_level: LogLevel = "INFO"):
    transforms = []
    transforms.append(PrefilterGenes(min_cells=3))
    transforms.append(HighlyVariableGenesRawCount(n_top_genes=1000))
    transforms.append(NormalizeTotalLog1P())
    transforms.append(CalSpatialNet(rad_cutoff=150))
    transforms.append(SetConfig({"label_channel": "label", "label_channel_type": "obs"}))
    return Compose(*transforms, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--sample_number", type=str, default="151673",
                        help="12 human dorsolateral prefrontal cortex datasets for the spatial domain task.")
    parser.add_argument("--n_components", type=int, default=50, help="Number of PC components.")
    parser.add_argument("--neighbors", type=int, default=17, help="Number of neighbors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--k", type=int, default=0, help="Number of clusters.")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--method", type=str, default="louvain", choices=["kmeans", "louvain"],
                        help="Clustering method.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu', 'cuda:0').")
    parser.add_argument("--obs_nums", type=int, default=10000)
    args = parser.parse_args()

    inner_scores = []
    scores = []
    times = []  # 新增：用于记录每次运行的时间

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间

        set_seed(seed)

        # Initialize model and get model specific preprocessing pipeline
        model = SpaGRA(k=args.k, n_epochs=args.n_epochs, method=args.method, random_seed=seed)
        preprocessing_pipeline = get_preprocessing_pipeline()

        # Load data and perform necessary preprocessing
        dataloader = SpatialLIBDDataset(data_id=args.sample_number)
        data = dataloader.load_data(transform=None, cache=args.cache)
        sub_data(data.data, args.obs_nums)
        preprocessing_pipeline(data)
        # Fit the model and evaluate
        model.fit(data.data)
        x, y = data.get_data(return_type="default")
        x = x.toarray()
        score = model.score(None, y.values)
        pred = model.predict()
        silhouette_score = resolve_score_func("silhouette")
        calinski_harabasz_score = resolve_score_func("calinski_harabasz")
        davies_bouldin_score = resolve_score_func("davies_bouldin")
        inner_scores.append(
            calculate_unified_scores({
                "silhouette": silhouette_score(x, pred),
                "calinski_harabasz": calinski_harabasz_score(x, pred),
                "davies_bouldin": davies_bouldin_score(x, pred)
            }))
        scores.append(score)

        end_time = time.time()  # 新增：记录单次循环的结束时间
        run_time = end_time - start_time
        times.append(run_time)  # 新增：保存耗时

        print(f"ARI: {score:.4f}, time: {run_time:.2f}s")  # 修改：打印单次得分与时间

    # 删除了原代码中这里重复的 print(f"ARI: {score:.4f}")
    print(f"spaGRA {args.sample_number}:")
    # 修改：加入 times 列表，以供 evaluator 捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间
""" To reproduce SpaGRA on other samples, please refer to command lines belows:

human dorsolateral prefrontal cortex sample 151673:
$ python spaGRA.py --sample_number 151673

human dorsolateral prefrontal cortex sample 151676:
$ python spaGRA.py --sample_number 151676

human dorsolateral prefrontal cortex sample 151507:
$ python spaGRA.py --sample_number 151507
"""
