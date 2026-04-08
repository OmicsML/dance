import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import bbknn
import numpy as np
import scanpy as sc

from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform

# 假设 BaseTransform 已定义
# logger = logging.getLogger(__name__)
LogLevel = Union[int, str]
Data = Any


@register_preprocessor("graph", "cell")
class BBKNNConstruction(BaseTransform):
    """Constructs a graph using BBKNN (for multi-batch data) or KNN (for single-batch
    data), trims edges based on ratio, and saves the graph structure as a list of
    strings in `adata.uns`.

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

    def __init__(self, batch_key: Optional[str] = None, edge_ratio: float = 2.0, key_added: str = 'temp_graph',
                 n_neighbors: int = 15, out: Optional[str] = None, log_level: LogLevel = "WARNING"):
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

            # 添加所有 KNN 边
            for i in range(len(rows)):
                temp_graph.append(f"{rows[i]} {cols[i]}")

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

            for i in range(len(rows)):
                b_row = batch_indices[rows[i]]
                b_col = batch_indices[cols[i]]

                if b_row != b_col:
                    x, y = max(b_row, b_col), min(b_row, b_col)
                    inter_ratio[x][y].append(ratio_values[i])

            # 排序
            for i in range(batch_num):
                for j in range(batch_num):
                    if len(inter_ratio[i][j]) > 0:
                        inter_ratio[i][j].sort(reverse=True)

            # 6. 生成 temp_graph 列表 (直接生成字符串)
            self.logger.info(f"Generating graph string list (Ratio: {self.edge_ratio})...")

            # 初始化列表，第一行为节点数
            temp_graph = [str(cell_count)]

            kept_edges_count = 0

            # 遍历边并应用筛选逻辑
            for i in range(len(rows)):
                b_row = batch_indices[rows[i]]
                b_col = batch_indices[cols[i]]

                keep_edge = False

                if b_row == b_col:
                    # 同批次：直接保留
                    keep_edge = True
                else:
                    # 跨批次：判断阈值
                    x, y = max(b_row, b_col), min(b_row, b_col)

                    # 计算截断阈值
                    limit_idx = int(self.edge_ratio * max(batch_info[x], batch_info[y]))
                    threshold_index = min(limit_idx, len(inter_ratio[x][y]) - 1)

                    ratio_threshold = inter_ratio[x][y][threshold_index]

                    if ratio_values[i] > ratio_threshold:
                        keep_edge = True

                if keep_edge:
                    # 直接格式化为 "row col" 字符串
                    temp_graph.append(f"{rows[i]} {cols[i]}")
                    kept_edges_count += 1

        # 7. 添加自环 (Self-loops)
        # 根据你的需求，显式添加 i i
        for index in range(cell_count):
            temp_graph.append(f"{index} {index}")

        self.logger.info(f"Graph list generated. Kept {kept_edges_count} edges + {cell_count} self-loops.")

        # 8. 存入 adata.uns
        # 注意：这可能会占用一定内存，但对于传递给 C++ 接口非常方便
        adata.uns[self.key_added] = temp_graph

        return data
