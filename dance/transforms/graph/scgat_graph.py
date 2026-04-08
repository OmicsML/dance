import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyGData

from dance import logger
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import LogLevel


def scipysparse2torchsparse(x):
    """Convert scipy sparse matrix to torch sparse tensor."""
    samples = x.shape[0]
    features = x.shape[1]
    values = x.data
    coo_data = x.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return indices, t


@register_preprocessor("graph", "cell")
class scGATGraphTransform(BaseTransform):
    """Constructs a PyTorch Geometric Graph from AnnData for GAT training.

    Supports labels in adata.obs (categorical) or adata.obsm (one-hot).

    """

    _DISPLAY_ATTRS: Tuple[str] = ("label_column", "n_neighbors")

    def __init__(self, label_column: str = 'cell_type', n_neighbors: int = 15, out: Optional[str] = None,
                 log_level: LogLevel = "INFO"):
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
                le.fit(le.classes_)  # Dummy fit to ensure compatibility
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
                val_idx_loc, test_idx_loc = train_test_split(test_idx_loc, test_size=0.5, random_state=42,
                                                             stratify=labels[test_idx_loc])
                val_indices = np.zeros(len(adata), dtype=bool)
                val_indices[val_idx_loc] = True
                test_indices = np.zeros(len(adata), dtype=bool)
                test_indices[test_idx_loc] = True
        else:
            # Random stratified split
            idx_all = np.arange(adata.shape[0])
            idx_train, idx_test = train_test_split(idx_all, test_size=1 - self.train_ratio, random_state=42,
                                                   stratify=labels)

            remaining_ratio = 1 - self.train_ratio
            if remaining_ratio > 0:
                val_relative_size = self.val_ratio / remaining_ratio
                idx_test, idx_val = train_test_split(idx_test, test_size=val_relative_size, random_state=42,
                                                     stratify=labels[idx_test])
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

        pyg_data = PyGData(x=torch.from_numpy(features).float(), edge_index=edge_index, y=torch.LongTensor(labels),
                           train_mask=torch.tensor(train_indices, dtype=torch.bool),
                           val_mask=torch.tensor(val_indices, dtype=torch.bool),
                           test_mask=torch.tensor(test_indices, dtype=torch.bool))

        self.logger.info(f"GAT Graph ready. Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
        self.logger.info(
            f"Split - Train: {pyg_data.train_mask.sum()}, Val: {pyg_data.val_mask.sum()}, Test: {pyg_data.test_mask.sum()}"
        )

        # Attach results to the dance Data object
        adata.uns['pyg_data'] = pyg_data
        adata.uns['label_encoder'] = le

        # Store masks in obs for verification
        adata.obs['gat_train_mask'] = train_indices
        adata.obs['gat_val_mask'] = val_indices
        adata.obs['gat_test_mask'] = test_indices

        self.logger.info(f"Transformation finished in {time.time() - start_time:.2f}s")

        return data
