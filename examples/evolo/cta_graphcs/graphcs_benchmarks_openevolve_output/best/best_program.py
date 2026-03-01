import argparse
import pprint
from typing import get_args

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import math
from dance import logger
from dance.data import Data
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.graphcs import GraphCSClassifier, load_GBP_data
from dance.transforms.filter import HighlyVariableGenesLogarithmizedByTopGenes, SupervisedFeatureSelection
from dance.transforms.graph.graphcs import BBKNNConstruction
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import NormalizeTotalLog1P
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
import scanpy as sc
import bbknn
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Any

from dance.transforms.base import BaseTransform
from dance.registry import register_preprocessor


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell",overwrite=True)
class BBKNNConstruction(BaseTransform):
    """
    Constructs a graph using BBKNN (for multi-batch data) or KNN (for single-batch data),
    trims edges based on ratio, and saves the graph structure as a list of strings in `adata.uns`.

    For multi-batch data, uses BBKNN to handle batch effects. For single-batch data,
    uses standard KNN neighbor graph construction.

    The format stored in `adata.uns[key_added]` is:
    [
      "cell_count",
      "row col",
      "row col",
      ...
      "index index" (self-loops)
    ]

    Parameters
    ----------
    batch_key
        The column name in adata.obs containing batch information. If None, assumes single-batch data.
    edge_ratio
        Parameter for controlling the number of inter-edges between batches (only used for multi-batch).
    key_added
        The key in `adata.uns` where the graph list will be stored. Default: 'temp_graph'.
    n_neighbors
        Number of neighbors for KNN (used for single-batch data). Default: 15.
    """

    _DISPLAY_ATTRS: Tuple[str] = ("batch_key", "edge_ratio", "key_added", "n_neighbors")

    def __init__(
        self,
        batch_key: Optional[str] = None,
        edge_ratio: float = 2.0,
        key_added: str = 'temp_graph',
        n_neighbors: int = 15,
        out: Optional[str] = None,
        log_level: LogLevel = "WARNING"
    ):
        super().__init__(out=out, log_level=log_level)
        self.batch_key = batch_key
        self.edge_ratio = edge_ratio
        self.key_added = key_added
        self.n_neighbors = n_neighbors

    def __call__(self, data: Data) -> Data:
        # 1. 解包 data
        adata = getattr(data, "data", data)

        if self.batch_key is None:
            # 单批次数据：使用标准 KNN
            self.logger.info("Starting single-batch graph construction using KNN")

            # 执行 PCA 和邻域图构建
            self.logger.info("Running KNN pipeline...")
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep='X_pca')

            # 提取连接信息
            if 'connectivities' not in adata.obsp:
                raise ValueError("KNN did not generate 'connectivities' in adata.obsp")

            cell_count = adata.shape[0]
            graph_mtx = adata.obsp['connectivities'].tocoo()

            rows = graph_mtx.row
            cols = graph_mtx.col

            # 对于单批次数据，保留所有 KNN 边
            self.logger.info("Generating graph string list for single-batch data...")

            # 初始化列表，第一行为节点数
            temp_graph = [str(cell_count)]

            kept_edges_count = len(rows)

            # 使用高度优化的向量化操作生成边字符串
            edge_strings = [f"{r} {c}" for r, c in zip(rows, cols)]
            temp_graph.extend(edge_strings)

        else:
            # 多批次数据：使用 BBKNN
            self.logger.info(f"Starting multi-batch graph construction. Batch key: '{self.batch_key}'")

            if self.batch_key not in adata.obs:
                raise ValueError(f"Batch key '{self.batch_key}' not found in data.obs.")

            # 2. 准备 Batch 信息
            unique_batches = adata.obs[self.batch_key].unique()
            batch_map = {b: i for i, b in enumerate(unique_batches)}

            # 生成整数型的 batch 数组 (n_cells, )
            batch_indices = adata.obs[self.batch_key].map(batch_map).values.astype(int)

            # 计算 batch_info (每个 batch 的细胞数量)
            batch_num = len(unique_batches)
            batch_info = [np.sum(batch_indices == i) for i in range(batch_num)]

            self.logger.info(f"Batches detected: {batch_num}, Counts: {batch_info}")

            # 3. 执行 BBKNN 流水线 (In-place)
            self.logger.info("Running BBKNN pipeline...")
            sc.tl.pca(adata)
            bbknn.bbknn(adata, batch_key=self.batch_key)
            sc.tl.leiden(adata, resolution=0.4)
            bbknn.ridge_regression(adata, batch_key=[self.batch_key], confounder_key=['leiden'])
            sc.pp.pca(adata)
            bbknn.bbknn(adata, batch_key=self.batch_key)

            # 4. 提取原始连接信息
            if 'connectivities' not in adata.obsp:
                 raise ValueError("BBKNN did not generate 'connectivities' in adata.obsp")

            cell_count = adata.shape[0]
            # 获取 COO 格式以便遍历
            graph_mtx = adata.obsp['connectivities'].tocoo()

            rows = graph_mtx.row
            cols = graph_mtx.col
            ratio_values = graph_mtx.data

            # 5. 统计边权重用于计算阈值
            inter_ratio = [[[] for _ in range(batch_num)] for _ in range(batch_num)]

            # 向量化统计跨批次边权重
            b_rows = batch_indices[rows]
            b_cols = batch_indices[cols]
            cross_batch_mask = b_rows != b_cols
            
            # 向量化收集跨批次边权重
            cross_batch_indices = np.where(cross_batch_mask)[0]
            for idx in cross_batch_indices:
                b_row = b_rows[idx]
                b_col = b_cols[idx]
                x, y = max(b_row, b_col), min(b_row, b_col)
                inter_ratio[x][y].append(ratio_values[idx])

            # 排序
            for i in range(batch_num):
                for j in range(batch_num):
                    if len(inter_ratio[i][j]) > 0:
                        inter_ratio[i][j].sort(reverse=True)

            # 6. 生成 temp_graph 列表 (直接生成字符串)
            self.logger.info(f"Generating graph string list (Ratio: {self.edge_ratio})...")

            # 初始化列表，第一行为节点数
            temp_graph = [str(cell_count)]

            # 向量化处理边的筛选逻辑
            # 同批次边直接保留
            same_batch_mask = b_rows == b_cols
            
            # 处理跨批次边的筛选 with improved efficiency
            filtered_cross_batch_indices = []
            
            # Precompute thresholds for each batch pair to avoid repeated calculations
            thresholds = {}
            for i in range(batch_num):
                for j in range(i):  # Only compute for upper triangle since it's symmetric
                    if len(inter_ratio[i][j]) > 0:
                        limit_idx = int(self.edge_ratio * max(batch_info[i], batch_info[j]))
                        threshold_index = min(limit_idx, len(inter_ratio[i][j]) - 1)
                        thresholds[(i, j)] = inter_ratio[i][j][threshold_index]
                        thresholds[(j, i)] = inter_ratio[i][j][threshold_index]  # Symmetric
            
            # Apply threshold filtering to cross-batch edges
            for idx in cross_batch_indices:
                b_row = b_rows[idx]
                b_col = b_cols[idx]
                
                # Determine the batch pair key
                x, y = max(b_row, b_col), min(b_row, b_col)
                threshold_key = (x, y)
                
                if threshold_key in thresholds and ratio_values[idx] > thresholds[threshold_key]:
                    filtered_cross_batch_indices.append(idx)

            # 构建最终的边索引数组
            final_edge_indices = np.concatenate([
                np.where(same_batch_mask)[0],
                np.array(filtered_cross_batch_indices)
            ])
            
            kept_edges_count = len(final_edge_indices)

            # 使用向量化操作生成保留边的字符串
            kept_rows = rows[final_edge_indices]
            kept_cols = cols[final_edge_indices]
            edge_strings = [f"{r} {c}" for r, c in zip(kept_rows, kept_cols)]
            temp_graph.extend(edge_strings)

        # 7. 添加自环 (Self-loops) - 使用向量化方法
        # 根据你的需求，显式添加 i i
        self_loop_strings = [f"{i} {i}" for i in range(cell_count)]
        temp_graph.extend(self_loop_strings)

        self.logger.info(f"Graph list generated. Kept {kept_edges_count} edges + {cell_count} self-loops.")

        # 8. 存入 adata.uns
        # 注意：这可能会占用一定内存，但对于传递给 C++ 接口非常方便
        adata.uns[self.key_added] = temp_graph

        return data

# EVOLVE-BLOCK-END

def get_preprocessing_pipeline(edge_ratio: float = 2, log_level: LogLevel = "INFO"):
        transforms = []
        transforms.append(SupervisedFeatureSelection(label_col="cell_type", n_features=2000,split_name="train"))
        transforms.append(NormalizeTotalLog1P())
        transforms.append(HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=2000))
        transforms.append(BBKNNConstruction(edge_ratio=edge_ratio, key_added="temp_graph"))
        transforms.append(SetConfig({
            "label_channel": "cell_type"
        }))
        return Compose(*transforms, log_level=log_level)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Base Dance arguments
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, set to -1 for CPU")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    
    # Training parameters (passed to GraphCSClassifier.__init__)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--vat_lr", type=float, default=0.1, help="VAT learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    
    # Model architecture parameters
    parser.add_argument("--layer", type=int, default=2, help="number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="hidden dimensions")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--bias", default='none', help="bias usage")
    
    # GraphCS specific parameters (if used in preprocessing or internal logic)
    parser.add_argument("--alpha", type=float, default=0.05, help="decay factor (GraphCS)")
    parser.add_argument("--rmax", type=float, default=1e-5, help="threshold (GraphCS)")
    parser.add_argument("--rrz", type=float, default=0.5, help="gamma/rrz (GraphCS)")
    parser.add_argument("--obs_nums",type=int,default=None)

    args = parser.parse_args()
    
    # Update GPU argument for the model wrapper expectations
    # The class expects args.gpus to be a list
    args.gpus = [args.gpu] if args.gpu != -1 else []
    
    logger.setLevel(args.log_level)
    logger.info(f"Running GraphCS with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    inner_scores=[]
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)
        
        # Initialize model: args are passed here, so self.batch_size etc are set now
        model = GraphCSClassifier(args, random_state=seed)
        
        # Get preprocessing pipeline
        preprocessing_pipeline = get_preprocessing_pipeline(log_level=args.log_level)
        
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        
        data = dataloader.load_data(transform=None, cache=args.cache)
        if args.obs_nums is not None:
            sub_data(data.data,args.obs_nums)
            train_idx, test_idx = train_test_split(range(args.obs_nums),test_size=0.2,random_state=args.seed)
            train_idx,val_idx = train_test_split(train_idx,test_size=args.val_size,random_state=args.seed)
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)
            data.set_split_idx("val", val_idx)
        preprocessing_pipeline(data)
        X,y=data.get_data(return_type="torch")
        temp_graph=data.data.uns["temp_graph"]
        dataset_name=f"{args.species}_{args.tissue}"
        features=load_GBP_data(dataset_name, args.alpha, args.rmax, args.rrz, X, temp_graph)
        # Obtain training and testing data
        x_train=features[data.train_idx]
        x_val=features[data.val_idx]
        x_test=features[data.test_idx]
        y_train=y[data.train_idx]
        y_val=y[data.val_idx]
        y_test=y[data.test_idx]
        
        # Convert OneHot labels to Integer labels for PyTorch CrossEntropy
        if y_train.shape[1] > 1:
            y_train_converted = y_train.argmax(1)
        else:
            y_train_converted = y_train.flatten()
        
        if y_val.shape[1] > 1:
            y_val_converted = y_val.argmax(1)
        else:
            y_val_converted = y_val.flatten()
            
        nfeat = x_train.shape[1]
        nclass = max(int(y_train_converted.max()),int(y_val_converted.max())) + 1
        # Train: fit() uses self.batch_size initialized earlier
        model.fit(torch.FloatTensor(x_train),torch.LongTensor(y_train_converted),torch.FloatTensor(x_val),torch.LongTensor(y_val_converted),nfeat,nclass)
        
        # Predict/Score: uses self.batch_size
        inner_score=model.score(x_val,y_val)
        score = model.score(x_test, y_test)
        inner_scores.append(inner_score)
        scores.append(score)
        print(f"{score=:.4f}")

    print(f"GraphCS {args.species} {args.tissue} {args.test_dataset}:")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(inner_scores)
    print(f"mean_score: {mean_score:.5f} +/- {std_score:.5f}")
    print(f"mean_inner_score: {mean_inner_score:.5f} +/- {std_inner_score:.5f}")

"""To reproduce GraphCS benchmarks, please refer to command lines below:

Mouse Brain
$ python graphcs.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lr 0.001 --hidden 256

Mouse Spleen
$ python graphcs.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --vat_lr 0.1

Mouse Kidney
$ python graphcs.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

$ python graphcs.py --species human --tissue Brain --train_dataset 328 --test_dataset 138
"""