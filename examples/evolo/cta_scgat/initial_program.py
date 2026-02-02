import argparse
import pprint
import numpy as np
import torch
import scanpy as sc

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scgat import scGATAnnotator
from dance.transforms import Compose, SetConfig
from dance.utils import set_seed

import hashlib
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data as PyGData
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod

import logging
from dance import logger
from dance.registry import register_preprocessor
from dance.typing import LogLevel
from dance.transforms.base import BaseTransform


# EVOLVE-BLOCK-START
def scipysparse2torchsparse(x):
    """Convert scipy sparse matrix to torch sparse tensor."""
    samples = x.shape[0]
    features = x.shape[1]
    values = x.data
    coo_data = x.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return indices, t

@register_preprocessor("graph", "cell",overwrite=True)
class scGATGraphTransform(BaseTransform):
    """
    Constructs a PyTorch Geometric Graph from AnnData for GAT training.
    Supports labels in adata.obs (categorical) or adata.obsm (one-hot).
    """

    _DISPLAY_ATTRS: Tuple[str] = ("label_column", "n_neighbors")

    def __init__(
        self,
        label_column: str = 'cell_type',
        n_neighbors: int = 15,
        out: Optional[str] = None,
        log_level: LogLevel = "INFO"
    ):
        super().__init__(out=out, log_level=log_level)
        self.label_column = label_column
        self.n_neighbors = n_neighbors

    def __call__(self, data) -> Any:
        self.train_indices = data.train_idx
        self.val_indices = data.val_idx
        self.test_indices = data.test_idx
        # Unpack AnnData
        if hasattr(data, "data") and isinstance(data.data, sc.AnnData):
            adata = data.data
        elif isinstance(data, sc.AnnData):
            adata = data
        else:
            raise TypeError(f"Input must be dance.Data or scanpy.AnnData, got {type(data)}")

        self.logger.info("Starting GAT Graph Construction...")
        start_time = time.time()

        # 1. Compute Neighbors if missing
        if 'neighbors' not in adata.uns:
            self.logger.info(f"Computing neighbors (k={self.n_neighbors})...")
            # 优先使用 PCA，如果没有则使用原始特征
            use_rep = 'X_pca' if 'X_pca' in adata.obsm else 'X'
            sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=use_rep)

        # 2. Process Adjacency Matrix
        self.logger.info("Processing adjacency matrix...")
        if 'connectivities' in adata.uns['neighbors']:
            adj = adata.uns['neighbors']['connectivities']
        elif 'connectivities_key' in adata.uns['neighbors']:
            adj_key = adata.uns['neighbors']['connectivities_key']
            adj = adata.obsp[adj_key]
        else:
            raise ValueError("Could not find connectivities matrix in neighbors")

        # Add self-loops (A_hat = A + I)
        adj = adj + sparse.diags([1] * adata.shape[0]).tocsr()

        # 3. Feature Normalization (Min-Max)
        self.logger.info("Normalizing features (MinMax)...")
        if sparse.issparse(adata.X):
            X_mat = adata.X.todense()
        else:
            X_mat = adata.X
        
        # Calculate min/max
        x_min = X_mat.min()
        x_max = X_mat.max()
        if x_max - x_min == 0:
            features = X_mat - x_min
        else:
            features = (X_mat - x_min) / (x_max - x_min)

        # ---------------------------------------------------------
        # 4. Handle Labels (Modified for One-Hot in obsm)
        # ---------------------------------------------------------
        self.logger.info(f"Processing label column: {self.label_column}")
        
        le = LabelEncoder()
        labels = None

        # Case A: Labels are in obsm (One-Hot Encoded)
        if self.label_column in adata.obsm:
            self.logger.info(f"Found '{self.label_column}' in adata.obsm (One-Hot format).")
            y_matrix = adata.obsm[self.label_column]
            
            # Check if it's a DataFrame (has column names) or numpy array
            if hasattr(y_matrix, "columns") or isinstance(y_matrix, pd.DataFrame):
                # Convert from One-Hot to Index (0, 1, 2...)
                labels = y_matrix.values.argmax(axis=1)
                # Store class names manually in encoder
                le.classes_ = np.array(y_matrix.columns)
                le.fit(le.classes_) # Dummy fit to ensure compatibility
            else:
                # Numpy array without names
                labels = np.array(y_matrix).argmax(axis=1)
                # Dummy classes 0, 1, 2...
                le.fit(np.arange(y_matrix.shape[1]))

        # Case B: Labels are in obs (Categorical column)
        elif self.label_column in adata.obs.columns:
            self.logger.info(f"Found '{self.label_column}' in adata.obs (Categorical format).")
            labels = le.fit_transform(adata.obs[self.label_column])
        
        else:
            raise ValueError(f"Label '{self.label_column}' not found in adata.obsm or adata.obs")

        self.logger.info(f"Encoded {len(le.classes_)} classes: {le.classes_}")

        # ---------------------------------------------------------
        # 5. Create Train/Val/Test Masks
        # ---------------------------------------------------------
        self.logger.info("Creating train/val/test splits...")

        # Check if indices are provided directly
        if self.train_indices is not None or self.val_indices is not None or self.test_indices is not None:
            self.logger.info("Using provided train/val/test indices...")
            n_cells = len(adata)

            # Convert indices to boolean masks
            train_mask = np.zeros(n_cells, dtype=bool)
            val_mask = np.zeros(n_cells, dtype=bool)
            test_mask = np.zeros(n_cells, dtype=bool)

            if self.train_indices is not None:
                train_mask[self.train_indices] = True
            if self.val_indices is not None:
                val_mask[self.val_indices] = True
            if self.test_indices is not None:
                test_mask[self.test_indices] = True

            train_indices = train_mask
            val_indices = val_mask
            test_indices = test_mask

        elif 'train_test_split' in adata.obs.columns:
            train_indices = (adata.obs['train_test_split'] == 'train').values
            test_indices = (adata.obs['train_test_split'] == 'test').values
            val_indices = (adata.obs['train_test_split'] == 'val').values

            if not val_indices.any():
                self.logger.info("Splitting test set to create validation set...")
                test_idx_loc = np.where(test_indices)[0]
                val_idx_loc, test_idx_loc = train_test_split(
                    test_idx_loc,
                    test_size=0.5,
                    random_state=42,
                    stratify=labels[test_idx_loc]
                )
                val_indices = np.zeros(len(adata), dtype=bool)
                val_indices[val_idx_loc] = True
                test_indices = np.zeros(len(adata), dtype=bool)
                test_indices[test_idx_loc] = True
        else:
            # Random stratified split
            idx_all = np.arange(adata.shape[0])
            idx_train, idx_test = train_test_split(
                idx_all,
                test_size=1 - self.train_ratio,
                random_state=42,
                stratify=labels
            )

            remaining_ratio = 1 - self.train_ratio
            if remaining_ratio > 0:
                val_relative_size = self.val_ratio / remaining_ratio
                idx_test, idx_val = train_test_split(
                    idx_test,
                    test_size=val_relative_size,
                    random_state=42,
                    stratify=labels[idx_test]
                )
            else:
                idx_val = []

            train_indices = np.zeros(len(adata), dtype=bool)
            val_indices = np.zeros(len(adata), dtype=bool)
            test_indices = np.zeros(len(adata), dtype=bool)

            train_indices[idx_train] = True
            val_indices[idx_val] = True
            test_indices[idx_test] = True

        # 6. Convert to PyTorch Geometric Data
        self.logger.info("Converting to PyG Data object...")
        edge_index, _ = scipysparse2torchsparse(adj)

        pyg_data = PyGData(
            x=torch.from_numpy(features).float(),
            edge_index=edge_index,
            y=torch.LongTensor(labels),
            train_mask=torch.tensor(train_indices, dtype=torch.bool),
            val_mask=torch.tensor(val_indices, dtype=torch.bool),
            test_mask=torch.tensor(test_indices, dtype=torch.bool)
        )

        self.logger.info(f"GAT Graph ready. Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
        self.logger.info(f"Split - Train: {pyg_data.train_mask.sum()}, Val: {pyg_data.val_mask.sum()}, Test: {pyg_data.test_mask.sum()}")

        # Attach results to the dance Data object
        adata.uns['pyg_data'] = pyg_data
        adata.uns['label_encoder'] = le
        
        # Store masks in obs for verification
        adata.obs['gat_train_mask'] = train_indices
        adata.obs['gat_val_mask'] = val_indices
        adata.obs['gat_test_mask'] = test_indices
        
        self.logger.info(f"Transformation finished in {time.time() - start_time:.2f}s")

        return data
# EVOLVE-BLOCK-END
def get_get_preprocessing_pipeline(label_column: str = 'cell_type',
                            n_neighbors: int = 15,log_level="INFO") -> BaseTransform:
    transforms=[]
    transforms.append(scGATGraphTransform(label_column=label_column,
                                n_neighbors=n_neighbors))
    transforms.append(SetConfig({
            "label_channel": "cell_type"
        }),)
    return Compose(*transforms, log_level=log_level)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    
    # GAT specific args
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5000)
    
    args = parser.parse_args()
    logger.setLevel("INFO")
    logger.info(f"Running GAT with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    inner_scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)
        
        # 1. 初始化模型 (参数需要显式传递，不能直接传 args)
        device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        model = scGATAnnotator(
            hidden_channels=args.hidden_channels,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=device,
            random_seed=seed
        )
        # 3. 加载并转换数据
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        data = dataloader.load_data(transform=None, cache=args.cache)
        
        # 2. 定义完整的预处理 Pipeline
        # 注意：scGATGraphTransform 只负责建图，特征提取(PCA)需要在此之前完成
        preprocessing_pipeline = get_get_preprocessing_pipeline(
                label_column="cell_type", # 确保这里与 dataset 的标签列名一致
                n_neighbors=15,
                log_level="INFO"
            )
        
        print(data)

        preprocessing_pipeline(data)

        # 4. 训练
        # 修改点：GAT 需要图结构，直接传入包含 'pyg_data' 的 AnnData 对象
        # data.data 是底层的 scanpy.AnnData 对象
        logger.info("Training GAT model...")
        
        # fit 内部会自动从 data.data.uns['pyg_data'] 提取数据
        model.fit(data.data)
        
        # 5. 评估
        logger.info("Evaluating...")
        
        # 获取真实标签用于计算准确率
        # get_test_data 返回的是 (feature, label) 元组，我们只需要 label
        _, y_val = data.get_val_data(return_type="torch")
        _, y_test = data.get_test_data(return_type="torch")
        if y_test.dim() > 1 and y_test.shape[1] > 1:
            y_test = y_test.argmax(1) # 转为 label index
        
        if y_val.dim() > 1 and y_val.shape[1] > 1:
            y_val = y_val.argmax(1) # 转为 label index
        # 进行预测
        # predict 会返回所有细胞的预测结果 (Array of shape [N_total])
        all_preds = model.predict(data.data)
        
        # 获取测试集掩码来提取对应的预测
        # scGATGraphTransform 会将 mask 存储在 pyg_data 中，也会同步到 obs 中 (gat_test_mask)
        # 或者我们可以直接利用 dance 数据集的划分逻辑
        # 最稳妥的方式是直接查看 pyg_data 中的 mask
        pyg_data = data.data.uns['pyg_data']
        test_mask = pyg_data.test_mask.cpu().numpy()
        val_mask = pyg_data.val_mask.cpu().numpy()
        # 提取测试集的预测结果
        y_pred = all_preds[test_mask]
        y_pred_val = all_preds[val_mask]
        # 计算准确率
        # 确保 y_pred 和 y_test 长度一致
        if len(y_pred) != len(y_test):
            logger.warning(f"Shape mismatch: Preds {len(y_pred)} vs Labels {len(y_test)}. "
                           "Using intersection or checking split logic.")
            # 这种情况通常极少发生，除非 cache 导致 split 不一致
        
        score = (y_pred == y_test.cpu().numpy()).mean()
        inner_score = (y_pred_val == y_val.cpu().numpy()).mean()
        scores.append(score)
        inner_scores.append(inner_score)
        print(f"{score=:.4f}")
        print(f"{inner_score=:.4f}")
    print(f"GAT {args.species} {args.tissue} {args.test_dataset}:")
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(inner_scores)
    print(f"mean_score: {mean_score:.5f} +/- {std_score:.5f}")
    print(f"mean_inner_score: {mean_inner_score:.5f} +/- {std_inner_score:.5f}")