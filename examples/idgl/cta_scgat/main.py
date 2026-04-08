import argparse
import copy
import hashlib
import json  # 新增：导入 json 模块用于保存结果
import logging
import pprint
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data as PyGData

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scgat import GAT, masked_nll_loss, scGATAnnotator
from dance.registry import register_preprocessor
from dance.transforms import Compose, SetConfig
from dance.transforms.base import BaseTransform
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data


class ChunkedGraphLearner(nn.Module):
    """Memory-efficient graph learner using chunked cosine similarity + KNN."""

    def __init__(self, n_feats: int, K: int = 20, lamb1: float = 0.5, lamb2: float = 0.5, chunk_size: int = 2000):
        super().__init__()
        self.transform = nn.Linear(n_feats, n_feats)
        self.K = K
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.chunk_size = chunk_size

    def forward(self, features: torch.Tensor, adj_init_sparse=None):
        z = torch.relu(self.transform(features))
        z_norm = F.normalize(z, p=2, dim=1, eps=1e-8)

        N = z_norm.shape[0]
        K = min(self.K, N - 1)

        all_src, all_dst, all_vals = [], [], []
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            chunk_len = end - i

            sim_chunk = z_norm[i:end] @ z_norm.T

            self_idx = torch.arange(chunk_len, device=features.device)
            diag_mask = torch.zeros_like(sim_chunk, dtype=torch.bool)
            diag_mask[self_idx, self_idx + i] = True
            sim_chunk = sim_chunk.masked_fill(diag_mask, -1.0)

            topk_vals, topk_idx = sim_chunk.topk(K, dim=1)
            topk_vals = torch.relu(topk_vals)
            rows = torch.arange(i, end, device=features.device).unsqueeze(1).expand(-1, K)
            all_src.append(rows.reshape(-1))
            all_dst.append(topk_idx.reshape(-1))
            all_vals.append(topk_vals.reshape(-1))

        knn_src = torch.cat(all_src)
        knn_dst = torch.cat(all_dst)
        knn_val = self.lamb1 * torch.cat(all_vals)

        if adj_init_sparse is not None:
            init_src, init_dst, init_val = adj_init_sparse
            src = torch.cat([knn_src, init_src])
            dst = torch.cat([knn_dst, init_dst])
            val = torch.cat([knn_val, self.lamb2 * torch.relu(init_val)])
        else:
            src, dst, val = knn_src, knn_dst, knn_val

        return src, dst, val


def smooth_cell_features(cell_feat: torch.Tensor, adj_sparse) -> torch.Tensor:
    """Smooth cell features using the learned adjacency (no gene nodes)."""
    if adj_sparse is None:
        return cell_feat

    knn_src, knn_dst, knn_val = adj_sparse
    N = cell_feat.shape[0]

    # Calculate row sums for normalization
    row_sums = torch.zeros(N, device=cell_feat.device).scatter_add(0, knn_src, knn_val).clamp(min=1e-8)
    row_sums_safe = row_sums.clone()
    row_sums_safe[row_sums_safe < 1e-8] = 1.0
    norm_val = knn_val / row_sums_safe[knn_src]

    # Use sparse matrix multiplication to avoid OOM
    indices = torch.stack([knn_src, knn_dst], dim=0)
    sparse_adj = torch.sparse_coo_tensor(indices, norm_val, size=(N, N))
    h_agg = torch.sparse.mm(sparse_adj, cell_feat)

    alpha = 0.5

    # 使用 In-place 操作替代 alpha * cell_feat + (1 - alpha) * h_agg
    # 这样可以避免分配额外的 4.5GB 显存
    h_agg.mul_(1.0 - alpha)
    h_agg.add_(cell_feat, alpha=alpha)

    return h_agg


def compute_graph_reg(knn_src: torch.Tensor, knn_dst: torch.Tensor, knn_val: torch.Tensor, cell_embeds: torch.Tensor,
                      N: int, lambda_smooth: float, lambda_conn: float, lambda_sparse: float,
                      chunk_size: int = 50000) -> torch.Tensor:
    """Compute full IDGL graph regularization loss (Smoothness + Connectivity +
    Sparsity).

    Smoothness is computed in chunks to save memory.

    """
    num_edges = knn_src.size(0)
    smoothness_loss = 0.0

    # 1. 平滑度损失 (Smoothness Loss) - 分块处理防止 OOM
    for i in range(0, num_edges, chunk_size):
        src_chunk = knn_src[i:i + chunk_size]
        dst_chunk = knn_dst[i:i + chunk_size]
        val_chunk = knn_val[i:i + chunk_size]

        diff = cell_embeds[src_chunk] - cell_embeds[dst_chunk]
        dist_sq = torch.sum(diff.pow_(2), dim=-1)
        smoothness_loss = smoothness_loss + torch.sum(val_chunk * dist_sq)

    # 2. 连通性损失 (Connectivity Loss) - 防止产生孤立节点
    row_sums = torch.zeros(N, device=knn_val.device).scatter_add(0, knn_src, knn_val)
    conn_loss = -torch.sum(torch.log(row_sums + 1e-8))

    # 3. 稀疏性损失 (Sparsity Loss) - 防止图过于密集
    sparse_loss = torch.sum(knn_val.pow(2))

    # 组合所有正则化项并取平均
    total_reg_loss = (lambda_smooth * smoothness_loss) + (lambda_conn * conn_loss) + (lambda_sparse * sparse_loss)

    return total_reg_loss / N


def scipysparse2torchsparse(x):
    """Convert scipy sparse matrix to torch sparse tensor."""
    values = x.data
    coo_data = x.tocoo()
    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [x.shape[0], x.shape[1]])
    return indices, t


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell", overwrite=True)
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
        if hasattr(data, "data") and isinstance(data.data, sc.AnnData):
            adata = data.data
        elif isinstance(data, sc.AnnData):
            adata = data
        else:
            raise TypeError(f"Input must be dance.Data or scanpy.AnnData, got {type(data)}")

        self.logger.info("Starting GAT Graph Construction...")
        start_time = time.time()

        if 'neighbors' not in adata.uns:
            self.logger.info(f"Computing neighbors (k={self.n_neighbors})...")
            use_rep = 'X_pca' if 'X_pca' in adata.obsm else 'X'
            sc.pp.neighbors(adata, n_neighbors=self.n_neighbors, use_rep=use_rep)

        self.logger.info("Processing adjacency matrix...")
        if 'connectivities' in adata.uns['neighbors']:
            adj = adata.uns['neighbors']['connectivities']
        elif 'connectivities_key' in adata.uns['neighbors']:
            adj_key = adata.uns['neighbors']['connectivities_key']
            adj = adata.obsp[adj_key]
        else:
            raise ValueError("Could not find connectivities matrix in neighbors")

        adj = adj + sparse.diags([1] * adata.shape[0]).tocsr()

        self.logger.info("Normalizing features (MinMax)...")
        if sparse.issparse(adata.X):
            X_mat = adata.X.todense()
        else:
            X_mat = adata.X

        x_min = X_mat.min()
        x_max = X_mat.max()
        if x_max - x_min == 0:
            features = X_mat - x_min
        else:
            features = (X_mat - x_min) / (x_max - x_min)

        self.logger.info(f"Processing label column: {self.label_column}")

        le = LabelEncoder()
        labels = None

        if self.label_column in adata.obsm:
            self.logger.info(f"Found '{self.label_column}' in adata.obsm (One-Hot format).")
            y_matrix = adata.obsm[self.label_column]
            if hasattr(y_matrix, "columns") or isinstance(y_matrix, pd.DataFrame):
                labels = y_matrix.values.argmax(axis=1)
                le.classes_ = np.array(y_matrix.columns)
                le.fit(le.classes_)
            else:
                labels = np.array(y_matrix).argmax(axis=1)
                le.fit(np.arange(y_matrix.shape[1]))

        elif self.label_column in adata.obs.columns:
            self.logger.info(f"Found '{self.label_column}' in adata.obs (Categorical format).")
            labels = le.fit_transform(adata.obs[self.label_column])

        else:
            raise ValueError(f"Label '{self.label_column}' not found in adata.obsm or adata.obs")

        self.logger.info(f"Encoded {len(le.classes_)} classes: {le.classes_}")

        self.logger.info("Creating train/val/test splits...")

        if self.train_indices is not None or self.val_indices is not None or self.test_indices is not None:
            self.logger.info("Using provided train/val/test indices...")
            n_cells = len(adata)

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

        adata.uns['pyg_data'] = pyg_data
        adata.uns['label_encoder'] = le

        adata.obs['gat_train_mask'] = train_indices
        adata.obs['gat_val_mask'] = val_indices
        adata.obs['gat_test_mask'] = test_indices

        self.logger.info(f"Transformation finished in {time.time() - start_time:.2f}s")

        return data


# EVOLVE-BLOCK-END


def get_preprocessing_pipeline(label_column: str = 'cell_type', n_neighbors: int = 15,
                               log_level="INFO") -> BaseTransform:
    transforms = []
    transforms.append(scGATGraphTransform(label_column=label_column, n_neighbors=n_neighbors))
    transforms.append(SetConfig({"label_channel": "cell_type"}))
    return Compose(*transforms, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int)
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--obs_nums", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.01)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--num_inner_iters", type=int, default=3)
    args = parser.parse_args()
    logger.setLevel("INFO")
    logger.info(f"Running GAT+IDGL with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    inner_scores = []
    times = []

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()
        set_seed(seed)

        device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"

        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        data = dataloader.load_data(transform=None, cache=args.cache)

        preprocessing_pipeline = get_preprocessing_pipeline(label_column="cell_type", n_neighbors=15, log_level="INFO")

        if args.obs_nums is not None:
            sub_data(data.data, args.obs_nums)
            train_idx, test_idx = train_test_split(range(args.obs_nums), test_size=0.2, random_state=args.seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=args.seed)
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)
            data.set_split_idx("val", val_idx)

        preprocessing_pipeline(data)

        pyg_data = data.data.uns['pyg_data']

        # Extract features and masks
        x_all = pyg_data.x.to(device)  # [N, F]
        y_all = pyg_data.y.to(device)  # [N]
        train_mask = pyg_data.train_mask.to(device)
        val_mask = pyg_data.val_mask.to(device)
        test_mask = pyg_data.test_mask.to(device)

        N, n_feats = x_all.shape
        n_classes = int(y_all.max().item() + 1)

        # Build initial sparse adj from PyG edge_index (unit weights)
        init_ei = pyg_data.edge_index.to(device)  # [2, E]
        adj_init_sparse = (
            init_ei[0],
            init_ei[1],
            torch.ones(init_ei.shape[1], device=device),
        )

        # Initialize models
        gat_model = GAT(
            in_channels=n_feats,
            hidden_channels=args.hidden_channels,
            out_channels=n_classes,
            heads=args.heads,
            dropout=args.dropout,
        ).to(device)

        graphlearner = ChunkedGraphLearner(n_feats, K=20, lamb1=0.5, lamb2=0.5, chunk_size=args.chunk_size).to(device)

        optimizer = torch.optim.Adagrad([
            {
                'params': gat_model.parameters()
            },
            {
                'params': graphlearner.parameters(),
                'lr': args.lr * 10
            },
        ], lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = float('inf')
        bad_counter = 0
        best_state = None
        cell_embeds = x_all.clone().detach()

        for epoch in range(args.n_epochs):
            gat_model.train()
            graphlearner.train()
            optimizer.zero_grad()

            # --- Inner iterations of GraphLearner ---
            for t in range(args.num_inner_iters):
                is_last = (t == args.num_inner_iters - 1)
                if is_last:
                    adj_cells_sparse = graphlearner(cell_embeds.detach(), adj_init_sparse)
                    feat_smooth = smooth_cell_features(x_all, adj_cells_sparse)
                else:
                    with torch.no_grad():
                        adj_t = graphlearner(cell_embeds, adj_init_sparse)
                        feat_t = smooth_cell_features(cell_embeds, adj_t)
                    cell_embeds = feat_t.detach()

            knn_src, knn_dst, knn_val = adj_cells_sparse

            # Regularization
            reg_loss = compute_graph_reg(knn_src, knn_dst, knn_val, cell_embeds.detach(), N, args.lambda_smooth,
                                         args.lambda_conn, args.lambda_sparse, chunk_size=args.chunk_size)

            # Build edge_index for GAT from learned adjacency
            learned_edge_index = torch.stack([knn_src, knn_dst], dim=0)

            # GAT forward pass
            output = gat_model(feat_smooth, learned_edge_index)

            cls_loss = masked_nll_loss(output, y_all, train_mask)
            total_loss = cls_loss + 1e-1 * reg_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(list(gat_model.parameters()) + list(graphlearner.parameters()), max_norm=2.0)
            optimizer.step()

            # Update cell_embeds
            gamma = 0.5
            cell_embeds = gamma * feat_smooth.detach() + (1 - gamma) * x_all

            # --- Validation ---
            gat_model.eval()
            graphlearner.eval()
            with torch.no_grad():
                val_output = gat_model(feat_smooth.detach(), learned_edge_index)
                if val_mask.sum() > 0:
                    val_loss = masked_nll_loss(val_output, y_all, val_mask).item()
                else:
                    val_loss = float('inf')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_counter = 0
                best_state = {
                    'gat': copy.deepcopy(gat_model.state_dict()),
                    'gl': copy.deepcopy(graphlearner.state_dict()),
                    'feat_smooth': feat_smooth.detach().clone(),
                    'edge_index': learned_edge_index.clone(),
                }
            else:
                bad_counter += 1

            if bad_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if epoch == 0 or (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1:04d} | Train Loss: {cls_loss.item():.4f} | "
                      f"Reg: {reg_loss.item():.4f} | Val Loss: {val_loss:.4f}")

        # --- Evaluation ---
        if best_state is not None:
            gat_model.load_state_dict(best_state['gat'])
            graphlearner.load_state_dict(best_state['gl'])
            best_feat = best_state['feat_smooth']
            best_ei = best_state['edge_index']
        else:
            best_feat = feat_smooth.detach()
            best_ei = learned_edge_index

        gat_model.eval()
        graphlearner.eval()
        with torch.no_grad():
            final_output = gat_model(best_feat, best_ei)
            preds = final_output.argmax(dim=1).cpu().numpy()

        y_np = y_all.cpu().numpy()
        score = (preds[test_mask.cpu().numpy()] == y_np[test_mask.cpu().numpy()]).mean()
        inner_score = (preds[train_mask.cpu().numpy()] == y_np[train_mask.cpu().numpy()]).mean()

        end_time = time.time()
        run_time = end_time - start_time

        scores.append(score)
        inner_scores.append(inner_score)
        times.append(run_time)

        print(f"Run {seed} | Score: {score:.4f}, Inner: {inner_score:.4f}, Time: {run_time:.2f}s")

    print(f"GAT+IDGL {args.species} {args.tissue} {args.test_dataset}:")
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(innerscores) if 'innerscores' in locals() else np.std(inner_scores)

    print(f"mean_score: {mean_score:.5f} +/- {std_score:.5f}")
    print(f"mean_inner_score: {mean_inner_score:.5f} +/- {std_inner_score:.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")

    # ==========================================
    # 新增：将结果保存到 JSON 文件
    # ==========================================
    results_dict = {
        "species": args.species,
        "tissue": args.tissue,
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "num_runs": args.num_runs,
        "scores": [float(s) for s in scores],
        "inner_scores": [float(s) for s in inner_scores],
        "times": [float(t) for t in times],
        "metrics": {
            "mean_score": float(mean_score),
            "std_score": float(std_score),
            "mean_inner_score": float(mean_inner_score),
            "std_inner_score": float(std_inner_score),
            "mean_time": float(np.mean(times))
        }
    }

    # 构造文件名，例如：results_mouse_Spleen_1759.json
    test_ds_str = "_".join(map(str, args.test_dataset))
    json_filename = f"results_{args.species}_{args.tissue}_{test_ds_str}.json"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    print(f"Results successfully saved to {json_filename}")
