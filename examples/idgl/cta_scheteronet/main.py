import argparse
import time
from typing import Optional
import json
import os  # 新增：导入 os 模块用于文件路径检查

import dgl
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scheteronet import (
    convert_dgl_to_original_format,
    eval_acc,
    print_statistics,
    scHeteroNet,
    set_graph_split,
    set_split,
)
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.filter import FilterCellsScanpy, FilterCellsType, HighlyVariableGenesLogarithmizedByTopGenes
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SaveRaw, SetConfig
from dance.transforms.normalize import Log1P, NormalizeTotal, UpdateSizeFactors
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data


class ChunkedGraphLearner(nn.Module):
    """Memory-efficient graph learner using chunked cosine similarity + KNN."""

    def __init__(self, n_feats: int, K: int = 20, lamb1: float = 0.5, lamb2: float = 0.5,
                 chunk_size: int = 2000):
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


# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell", overwrite=True)
class HeteronetGraph(BaseTransform):

    def __init__(self, knn_num: int = 5, distance_metrics: str = 'l2', random_state: int = 0,
                 channel: Optional[str] = None, channel_type: Optional[str] = "X", ignore_first: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.knn_num = knn_num
        self.distance_metrics = distance_metrics
        self.random_state = random_state
        self.channel = channel
        self.ignore_first = ignore_first
        self.channel_type = channel_type

    def build_graph(self, features_np, radius=None, knears=None, distance_metrics='l2'):
        coor = pd.DataFrame(features_np)
        if radius:
            nbrs = NearestNeighbors(radius=radius, metric=distance_metrics).fit(coor)
            _, indices = nbrs.radius_neighbors(coor, return_distance=True)
        else:
            nbrs = NearestNeighbors(n_neighbors=knears + 1, metric=distance_metrics).fit(coor)
            _, indices = nbrs.kneighbors(coor)

        edge_list = np.array([[i, j] for i, sublist in enumerate(indices) for j in sublist])
        return edge_list

    def __call__(self, data):
        adata = data.data
        features_np = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        features = torch.as_tensor(features_np, dtype=torch.float32)
        num_nodes = features.shape[0]

        labels_np = np.argmax(adata.obsm['cell_type'].copy(), axis=1)
        labels = torch.as_tensor(labels_np, dtype=torch.long)

        batchs = adata.obs.get('batch_id', None)

        if self.ignore_first:
            labels[labels == 0] = -1

        edge_list_np = self.build_graph(features_np, knears=self.knn_num, distance_metrics=self.distance_metrics)
        if edge_list_np.shape[0] == 0:
            src = torch.tensor([], dtype=torch.long)
            dst = torch.tensor([], dtype=torch.long)
        else:
            edge_list_tensor = torch.tensor(edge_list_np.T, dtype=torch.long)
            src, dst = edge_list_tensor[0], edge_list_tensor[1]

        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.ndata['feat'] = features
        g.ndata['label'] = labels
        if batchs is not None:
            g.ndata['batch_id'] = torch.from_numpy(batchs.values.astype(int)).long()
        adata.uns[self.out] = g
# EVOLVE-BLOCK-END


def smooth_cell_features(cell_feat, adj_sparse):
    """Smooth cell features using learned adjacency (no gene nodes, pure cell graph)."""
    if adj_sparse is None:
        return cell_feat
    knn_src, knn_dst, knn_val = adj_sparse
    N = cell_feat.shape[0]
    
    # 计算行归一化系数
    row_sums = torch.zeros(N, device=cell_feat.device).scatter_add(
        0, knn_src, knn_val).clamp(min=1e-8)
    row_sums_safe = row_sums.clone()
    row_sums_safe[row_sums_safe < 1e-8] = 1.0
    norm_val = knn_val / row_sums_safe[knn_src]

    # ========== 优化部分：使用稀疏矩阵乘法 (SpMM) ==========
    # 构建稀疏张量 (Sparse Tensor)
    indices = torch.stack([knn_src, knn_dst], dim=0)
    # 注意：如果存在重复的边，sparse_coo_tensor 会自动将它们的值相加（coalesce）
    adj_sparse_tensor = torch.sparse_coo_tensor(
        indices, norm_val, size=(N, N), device=cell_feat.device
    )
    
    # 使用稀疏矩阵乘法进行特征聚合，避免实例化 [E, D] 的巨大张量
    h_agg = torch.sparse.mm(adj_sparse_tensor, cell_feat)
    # ========================================================

    alpha = 0.5
    return alpha * cell_feat + (1 - alpha) * h_agg


def compute_graph_reg(knn_src, knn_dst, knn_val, cell_embeds, N,
                      lambda_smooth, lambda_conn, lambda_sparse):
    reg = torch.tensor(0.0, device=knn_val.device)

    if lambda_smooth > 0:
        diff = cell_embeds[knn_src] - cell_embeds[knn_dst]
        smooth_loss = (knn_val * (diff * diff).sum(dim=1)).mean()
        reg = reg + lambda_smooth * smooth_loss

    if lambda_conn > 0:
        row_sums = torch.zeros(N, device=knn_val.device).scatter_add(0, knn_src, knn_val)
        target_degree = row_sums.mean().detach()
        conn_loss = ((row_sums - target_degree) ** 2).mean()
        reg = reg + lambda_conn * conn_loss

    if lambda_sparse > 0:
        sparse_loss = (knn_val * knn_val).mean()
        reg = reg + lambda_sparse * sparse_loss

    return reg


def get_preprocessing_pipeline(log_level: LogLevel = "INFO"):
    transforms = []
    transforms.append(FilterCellsType())
    transforms.append(AnnDataTransform(sc.pp.filter_genes, min_counts=3))
    transforms.append(FilterCellsScanpy(min_counts=1))
    transforms.append(HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=4000, flavor="cell_ranger"))
    transforms.append(SaveRaw())
    transforms.append(NormalizeTotal())
    transforms.append(UpdateSizeFactors())
    transforms.append(Log1P())
    transforms.append(HeteronetGraph())
    transforms.append(SetConfig({"label_channel": "cell_type"}))
    return Compose(*transforms, log_level=log_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    parser.add_argument("--species", default="mouse", type=str)

    parser.add_argument('--data_dir', type=str, default='../temp_data')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_step', type=int, default=1, help='how often to print')
    parser.add_argument("--num_runs", type=int, default=5, help="Number of repetitions")
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers for deep methods')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of mlp layers')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodprop', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--oodalpha', type=float, default=0.3, help='weight for residual connection in propagation')
    parser.add_argument('--use_zinb', action='store_true',
                        help='whether to use ZINB loss (use if you do not need this)')
    parser.add_argument('--use_2hop', action='store_false',
                        help='whether to use 2-hop propagation (use if you do not need this)')
    parser.add_argument('--zinb_weight', type=float, default=1e-4)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    # display and utility
    parser.add_argument('--display_step', type=int, default=10, help='how often to print')
    parser.add_argument('--print_prop', action='store_true', help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true', help='print args for hyper-parameter searching')
    parser.add_argument('--cl_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--spatial', action='store_false', help='read spatial')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obs_nums", type=int, default=None)
    # IDGL-specific args
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.01)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--num_inner_iters", type=int, default=3)
    args = parser.parse_args()

    runs = args.num_runs
    results = []
    inner_scores = []
    times = []

    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    eval_func = eval_acc

    for run in range(runs):
        start_time = time.time()

        set_seed(args.seed + run)
        dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                               train_dataset=args.train_dataset, data_dir=args.data_dir,
                                               val_size=args.val_size)
        ref_data_name = f"{args.species}_{args.tissue}_{args.train_dataset}"
        preprocessing_pipeline = get_preprocessing_pipeline()
        data = dataloader.load_data(transform=None, cache=args.cache)
        if args.obs_nums is not None:
            sub_data(data.data, args.obs_nums)
            train_idx, test_idx = train_test_split(range(args.obs_nums), test_size=0.2, random_state=args.seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=args.val_size, random_state=args.seed)
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)
            data.set_split_idx("val", val_idx)
        preprocessing_pipeline(data)
        set_split(data, data.train_idx, data.val_idx, data.test_idx)

        g = data.data.uns['HeteronetGraph']
        dataset_ind, dataset_ood_tr, dataset_ood_te, adata = convert_dgl_to_original_format(g, data.data, ref_data_name)
        if len(dataset_ind.y.shape) == 1:
            dataset_ind.y = dataset_ind.y.unsqueeze(1)
        if len(dataset_ood_tr.y.shape) == 1:
            dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
        if isinstance(dataset_ood_te, list):
            for single_dataset_ood_te in dataset_ood_te:
                if len(single_dataset_ood_te.y.shape) == 1:
                    single_dataset_ood_te.y = single_dataset_ood_te.y.unsqueeze(1)
        else:
            if len(dataset_ood_te.y.shape) == 1:
                dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

        c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
        d = dataset_ind.graph['node_feat'].shape[1]
        model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes,
                            hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                            use_bn=args.use_bn, device=device, min_loss=100000)
        criterion = nn.NLLLoss()
        model.train()
        model.reset_parameters()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # --- IDGL Setup ---
        cell_feat_orig = dataset_ind.graph['node_feat'].to(device)
        N = cell_feat_orig.shape[0]

        # Extract initial adj from the KNN edge_index built by HeteronetGraph
        edge_index = dataset_ind.edge_index.to(device)
        adj_init_sparse = (edge_index[0], edge_index[1],
                           torch.ones(edge_index.shape[1], device=device))

        graphlearner = ChunkedGraphLearner(d, K=20, lamb1=0.5, lamb2=0.5,
                                           chunk_size=args.chunk_size).to(device)
        gl_optimizer = torch.optim.Adam(graphlearner.parameters(), lr=args.lr * 10)

        cell_embeds = cell_feat_orig.clone().detach()

        test_score = 0.0
        inner_score = 0.0

        for epoch in range(args.epochs):
            model.train()
            graphlearner.train()

            # --- 1. Inner iterations: iteratively refine graph and smooth features ---
            for t in range(args.num_inner_iters):
                is_last_iter = (t == args.num_inner_iters - 1)
                if is_last_iter:
                    adj_cells_sparse = graphlearner(cell_embeds.detach(), adj_init_sparse)
                else:
                    with torch.no_grad():
                        adj_t = graphlearner(cell_embeds, adj_init_sparse)
                        feat_t = smooth_cell_features(cell_embeds, adj_t)
                    cell_embeds = feat_t.detach()

            # --- 2. Compute graph regularization and update GraphLearner ---
            knn_src, knn_dst, knn_val = adj_cells_sparse
            reg_loss = compute_graph_reg(
                knn_src, knn_dst, knn_val, cell_embeds.detach(),
                N, args.lambda_smooth, args.lambda_conn, args.lambda_sparse,
            )
            gl_optimizer.zero_grad()
            reg_loss.backward()
            gl_optimizer.step()

            # --- 3. Smooth original features for GNN input ---
            with torch.no_grad():
                adj_smooth = graphlearner(cell_embeds, adj_init_sparse)
                smoothed_feat = smooth_cell_features(cell_feat_orig, adj_smooth)

            # Update dataset node features so model.fit uses smoothed features
            dataset_ind.graph['node_feat'] = smoothed_feat.cpu()

            # --- 4. Train GNN ---
            loss = model.fit(dataset_ind, dataset_ood_tr, args.use_zinb, adata, args.zinb_weight, args.cl_weight,
                             args.mask_ratio, criterion, optimizer)

            # --- 5. Update cell embeds ---
            gamma = 0.5
            cell_embeds = gamma * smoothed_feat.detach() + (1 - gamma) * cell_feat_orig

            if epoch == 0 or (epoch + 1) % args.display_step == 0:
                model.eval()
                graphlearner.eval()
                with torch.no_grad():
                    test_score = model.score(dataset_ind, dataset_ind.y, data.test_idx)
                    inner_score = model.score(dataset_ind, dataset_ind.y, data.train_idx)
                print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Reg: {reg_loss.item():.4f} "
                      f"| Test: {test_score:.4f} | Train: {inner_score:.4f}")

        results.append(test_score)
        inner_scores.append(inner_score)

        end_time = time.time()
        run_time = end_time - start_time
        times.append(run_time)

        print(f"Run {run + 1} - test_score: {test_score:.4f}, time: {run_time:.2f}s")

    print(f"scHeteroNet {args.species} {args.tissue} {args.test_dataset}:")
    print(f"scores:{results},inner_scores:{inner_scores},times:{times}")
    
    mean_score = np.mean(results)
    std_score = np.std(results)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(inner_scores)

    print(f"mean_score: {mean_score:.5f} +/- {std_score:.5f}")
    print(f"mean_inner_score: {mean_inner_score:.5f} +/- {std_inner_score:.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")

    # ==========================================
    # 统一追加保存到同一个 JSON 文件
    # ==========================================
    results_dict = {
        "species": args.species,
        "tissue": args.tissue,
        "train_dataset": args.train_dataset,
        "test_dataset": args.test_dataset,
        "num_runs": args.num_runs,
        "scores": [float(s) for s in results],
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

    json_filename = "results_scheteronet.json"
    
    # 读取已有的数据
    if os.path.exists(json_filename):
        with open(json_filename, "r", encoding="utf-8") as f:
            try:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = [all_results]
            except json.JSONDecodeError:
                all_results = []
    else:
        all_results = []

    # 追加当前实验结果
    all_results.append(results_dict)

    # 写回文件
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print(f"Results successfully appended to {json_filename}")