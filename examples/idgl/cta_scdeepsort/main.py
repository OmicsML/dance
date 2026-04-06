import argparse
import pprint
from typing import Optional, Union, get_args
import json  # 新增：导入 json 模块用于保存结果

import dgl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort, GNN
import torch.nn as nn
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.transforms.misc import Compose, SetConfig
from dance.typing import LogLevel
from dance.utils import set_seed, sub_data
from dance.utils.matrix import normalize
from dance.utils.wrappers import add_mod_and_transform
import time
import torch.nn.functional as F


class ChunkedGraphLearner(nn.Module):
    """Memory-efficient graph learner using chunked cosine similarity + KNN."""

    def __init__(self, n_feats: int, K: int = 20, lamb1: float = 0.5, lamb2: float = 0.5,
                 chunk_size: int = 2000):
        super().__init__()
        # 【修复 2】增强 Graph Learner 的表达能力，从单一的对角权重升级为 Linear 层
        self.transform = nn.Linear(n_feats, n_feats)
        self.K = K
        self.lamb1 = lamb1
        self.lamb2 = lamb2
        self.chunk_size = chunk_size

    def forward(self, features: torch.Tensor, adj_init_sparse=None):
        # 使用 Linear 层进行特征变换，并用 ReLU 激活
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
@register_preprocessor("feature", "cell",overwrite=True)
@add_mod_and_transform
class WeightedFeaturePCA(BaseTransform):
    _DISPLAY_ATTRS = ("n_components", "split_name", "feat_norm_mode", "feat_norm_axis")

    def __init__(self, n_components: Union[float, int] = 400, split_name: Optional[str] = None,
                 feat_norm_mode: Optional[str] = None, feat_norm_axis: int = 0, save_info=False, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.split_name = split_name
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.save_info = save_info

    def __call__(self, data):
        feat = data.get_x(self.split_name)  
        if self.feat_norm_mode is not None:
            feat = normalize(feat, mode=self.feat_norm_mode, axis=self.feat_norm_axis)
        if self.n_components > min(feat.shape):
            self.n_components = min(feat.shape)
        gene_pca = PCA(n_components=self.n_components)  

        gene_feat = gene_pca.fit_transform(feat.T)  

        x = data.get_x()
        cell_feat = normalize(x, mode="normalize", axis=1) @ gene_feat  
        data.data.obsm[self.out] = cell_feat.astype(np.float32)
        data.data.varm[self.out] = gene_feat.astype(np.float32)
        return data

@register_preprocessor("graph", "cell",overwrite=True)
class CellFeatureGraph(BaseTransform):
    def __init__(self, cell_feature_channel: str, gene_feature_channel: Optional[str] = None, *,
                 mod: Optional[str] = None, normalize_edges: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cell_feature_channel = cell_feature_channel
        self.gene_feature_channel = gene_feature_channel or cell_feature_channel
        self.mod = mod
        self.normalize_edges = normalize_edges

    def __call__(self, data):
        feat = data.get_feature(return_type="default", mod=self.mod)
        num_cells, num_feats = feat.shape

        row, col = np.nonzero(feat)
        edata = np.array(feat[row, col]).ravel()[:, None]

        row = row + num_feats  
        col, row = np.hstack((col, row)), np.hstack((row, col))  
        edata = np.vstack((edata, edata))

        col = torch.LongTensor(col)
        row = torch.LongTensor(row)
        edata = torch.FloatTensor(edata)

        g = dgl.graph((row, col))
        g.edata["weight"] = edata
        g.ndata["cell_id"] = torch.concat((torch.arange(num_feats, dtype=torch.int32),
                                           -torch.ones(num_cells, dtype=torch.int32)))  
        g.ndata["feat_id"] = torch.concat((-torch.ones(num_feats, dtype=torch.int32),
                                           torch.arange(num_cells, dtype=torch.int32)))  

        if self.normalize_edges:
            in_deg = g.in_degrees()
            for i in range(g.number_of_nodes()):
                src, dst, eidx = g.in_edges(i, form="all")
                if src.shape[0] > 0:
                    edge_w = g.edata["weight"][eidx]
                    g.edata["weight"][eidx] = in_deg[i] * edge_w / edge_w.sum()
        g.add_edges(g.nodes(), g.nodes(), {"weight": torch.ones(g.number_of_nodes())[:, None]})

        gene_feature = data.get_feature(return_type="torch", channel=self.gene_feature_channel, mod=self.mod,
                                        channel_type="varm")
        cell_feature = data.get_feature(return_type="torch", channel=self.cell_feature_channel, mod=self.mod,
                                        channel_type="obsm")
        g.ndata["features"] = torch.vstack((gene_feature, cell_feature))

        data.data.uns[self.out] = g
        return data

@register_preprocessor("graph", "cell",overwrite=True)
class PCACellFeatureGraph(BaseTransform):
    _DISPLAY_ATTRS = ("n_components", "split_name")

    def __init__(self, n_components: int = 400, split_name: Optional[str] = None, *,
                 normalize_edges: bool = True, feat_norm_mode: Optional[str] = None,
                 feat_norm_axis: int = 0, mod: Optional[str] = None, log_level: LogLevel = "WARNING"):
        super().__init__(log_level=log_level)
        self.n_components = n_components
        self.split_name = split_name
        self.normalize_edges = normalize_edges
        self.feat_norm_mode = feat_norm_mode
        self.feat_norm_axis = feat_norm_axis
        self.mod = mod

    def __call__(self, data):
        WeightedFeaturePCA(self.n_components, self.split_name, feat_norm_mode=self.feat_norm_mode,
                           feat_norm_axis=self.feat_norm_axis, log_level=self.log_level)(data)
        CellFeatureGraph(cell_feature_channel="WeightedFeaturePCA", mod=self.mod, normalize_edges=self.normalize_edges,
                         log_level=self.log_level)(data)
        return data
# EVOLVE-BLOCK-END


def smooth_cell_features(cell_feat, adj_sparse, gene_feat):
    if adj_sparse is None:
        return torch.cat([gene_feat, cell_feat], dim=0)
    knn_src, knn_dst, knn_val = adj_sparse
    N = cell_feat.shape[0]
    row_sums = torch.zeros(N, device=cell_feat.device).scatter_add(
        0, knn_src, knn_val).clamp(min=1e-8)
    row_sums_safe = row_sums.clone()
    row_sums_safe[row_sums_safe < 1e-8] = 1.0
    norm_val = knn_val / row_sums_safe[knn_src]
    
    h_agg = torch.zeros_like(cell_feat).scatter_add(
        0, knn_src.unsqueeze(-1).expand(-1, cell_feat.shape[1]),
        norm_val.unsqueeze(-1) * cell_feat[knn_dst])
    
    alpha = 0.5  
    smoothed_cell_feat = alpha * cell_feat + (1 - alpha) * h_agg
    
    return torch.cat([gene_feat, smoothed_cell_feat], dim=0)


def compute_graph_reg(knn_src, knn_dst, knn_val, cell_embeds, N,
                      lambda_smooth, lambda_conn, lambda_sparse):
    reg = torch.tensor(0.0, device=knn_val.device)

    if lambda_smooth > 0:
        diff = cell_embeds[knn_src] - cell_embeds[knn_dst]  
        smooth_loss = (knn_val * (diff * diff).sum(dim=1)).mean()
        reg = reg + lambda_smooth * smooth_loss

    if lambda_conn > 0:
        row_sums = torch.zeros(N, device=knn_val.device).scatter_add(0, knn_src, knn_val)
        # 【修复 1】不再强制逼近 1.0，而是逼近平均度数，惩罚度数的方差，鼓励均匀连通
        target_degree = row_sums.mean().detach()
        conn_loss = ((row_sums - target_degree) ** 2).mean()
        reg = reg + lambda_conn * conn_loss

    if lambda_sparse > 0:
        sparse_loss = (knn_val * knn_val).mean()
        reg = reg + lambda_sparse * sparse_loss

    return reg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--dense_dim", type=int, default=400)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759])
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970])
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--val_size", type=float, default=0.0)
    parser.add_argument("--obs_nums", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--lambda_smooth", type=float, default=0.1)
    parser.add_argument("--lambda_conn", type=float, default=0.01)
    parser.add_argument("--lambda_sparse", type=float, default=0.01)
    parser.add_argument("--num_inner_iters", type=int, default=3)
    parser.add_argument("--eps_adj", type=float, default=0.0)
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    scores = []
    inner_scores = []
    times = []

    for seed in range(args.seed, args.seed + args.num_runs):
        start_time = time.time()
        set_seed(seed)

        model = ScDeepSort(args.dense_dim, args.hidden_dim, args.n_layers, args.species, args.tissue,
                           dropout=args.dropout, batch_size=args.batch_size, device=args.device)
        preprocessing_pipeline = Compose(
            PCACellFeatureGraph(n_components=args.dense_dim, split_name="train"),
            SetConfig({"label_channel": "cell_type"}),
            log_level="INFO",
        )

        dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                               train_dataset=args.train_dataset, data_dir="../temp_data", val_size=args.val_size)
        data = dataloader.load_data(transform=None, cache=args.cache)

        if args.obs_nums is not None:
            sub_data(data.data, args.obs_nums)
            train_idx, test_idx = train_test_split(range(args.obs_nums), test_size=0.2, random_state=seed)
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)

        preprocessing_pipeline(data)

        y_train = data.get_y(split_name="train", return_type="torch").to(args.device)
        y_test = data.get_y(split_name="test", return_type="torch").to(args.device)
        labels_train = y_train.argmax(1)
        labels_test = y_test.argmax(1)

        g = data.data.uns["CellFeatureGraph"].to(args.device)
        num_genes = data.shape[1]
        gene_ids = torch.arange(num_genes, device=args.device)
        train_cell_ids = torch.LongTensor(data.train_idx).to(args.device) + num_genes
        test_cell_ids = torch.LongTensor(data.test_idx).to(args.device) + num_genes

        g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
        g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

        features_train = g_train.ndata["features"].to(args.device)
        n_feats = features_train.shape[1]
        num_train_cells = g_train.num_nodes() - num_genes

        cell_features_train = features_train[num_genes:]  
        gene_feat_base = features_train[:num_genes].detach()

        src_all, dst_all = g_train.edges()
        cell_mask = (src_all >= num_genes) & (dst_all >= num_genes)
        if cell_mask.any():
            weights_all = g_train.edata["weight"].squeeze(-1) if "weight" in g_train.edata \
                else torch.ones(src_all.shape[0], device=args.device)
            adj_init_sparse = (src_all[cell_mask] - num_genes,
                               dst_all[cell_mask] - num_genes,
                               weights_all[cell_mask])
        else:
            adj_init_sparse = None

        graphlearner = ChunkedGraphLearner(n_feats, K=20, lamb1=0.5, lamb2=0.5,
                                           chunk_size=args.chunk_size).to(args.device)

        model.num_labels = labels_train.max().item() + 1
        model.model = GNN(model.dense_dim, model.num_labels, model.hidden_dim, model.n_layers, num_genes,
                          activation=nn.ReLU(), dropout=model.dropout).to(args.device)

        optimizer = torch.optim.Adam([
            {'params': model.model.parameters()},
            {'params': graphlearner.parameters(), 'lr': args.lr * 10}
        ], lr=args.lr, weight_decay=args.weight_decay)

        g_train_cpu = g_train.to("cpu")
        g_test_cpu = g_test.to("cpu")
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.model.n_layers)
        num_test_cells = g_test.num_nodes() - num_genes
        train_loader = dgl.dataloading.DataLoader(
            g_train_cpu,
            torch.arange(num_genes, num_genes + num_train_cells),
            sampler, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        test_loader = dgl.dataloading.DataLoader(
            g_test_cpu,
            torch.arange(num_genes, num_genes + num_test_cells),
            sampler, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )

        model.model.train()
        graphlearner.train()

        train_batches = list(train_loader)
        num_batches = len(train_batches)
        cell_embeds = cell_features_train.clone().detach()

        for epoch in range(args.n_epochs):

            # --- 1. GraphLearner 前向传播与正则化计算 (全图) ---
            for t in range(args.num_inner_iters):
                is_last_iter = (t == args.num_inner_iters - 1)
                if is_last_iter:
                    adj_cells_sparse = graphlearner(cell_embeds.detach(), adj_init_sparse)
                    feat_smooth = smooth_cell_features(cell_features_train, adj_cells_sparse, gene_feat_base)
                else:
                    with torch.no_grad():
                        adj_t = graphlearner(cell_embeds, adj_init_sparse)
                        feat_t = smooth_cell_features(cell_embeds, adj_t, gene_feat_base)
                    cell_embeds = feat_t[num_genes:].detach()

            knn_src, knn_dst, knn_val = adj_cells_sparse
            reg_loss = compute_graph_reg(
                knn_src, knn_dst, knn_val, cell_embeds.detach(),
                num_train_cells,
                args.lambda_smooth, args.lambda_conn, args.lambda_sparse,
            )

            total_loss_val = 0.0
            epoch_hidden = torch.zeros(num_train_cells, args.hidden_dim, device=args.device)

            # --- 2. GNN Batch 训练 ---
            for batch_idx, (input_nodes, output_nodes, blocks) in enumerate(train_batches):
                optimizer.zero_grad()
                
                blocks_gpu = [b.to(args.device) for b in blocks]
                src_ids = blocks_gpu[0].srcdata[dgl.NID]
                batch_feats = feat_smooth[src_ids]

                x = batch_feats
                for block, layer in zip(blocks_gpu, model.model.layers):
                    x = layer(block, x)
                hidden = x
                batch_out = model.model.linear(hidden)

                batch_cell_idx = (output_nodes - num_genes).to(args.device)
                batch_loss = F.cross_entropy(batch_out, labels_train[batch_cell_idx])

                # 【修复核心】只在最后一个 batch 反向传播 reg_loss，避免使用过时梯度重复更新
                is_last_batch = (batch_idx == num_batches - 1)
                
                if is_last_batch:
                    alpha = 1e-1 
                    total_batch_loss = batch_loss + alpha * reg_loss
                    total_batch_loss.backward()
                else:
                    # 前面的 batch 只反向传播 GNN 的 loss，需要 retain_graph 以便最后传播 reg_loss
                    batch_loss.backward(retain_graph=True)
                
                total_loss_val += batch_loss.item()
                epoch_hidden[batch_cell_idx] = hidden.detach()
                
                all_params = list(model.model.parameters()) + list(graphlearner.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=2.0)
                
                optimizer.step()

            gamma = 0.5 
            cell_embeds = gamma * feat_smooth[num_genes:].detach() + (1 - gamma) * cell_features_train
            
            if epoch==0 or (epoch + 1) % 50 == 0:
                avg_loss = total_loss_val / num_batches
                print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Reg: {reg_loss.item():.4f}")

        # --- Evaluation ---
        model.model.eval()
        graphlearner.eval()
        with torch.no_grad():
            adj_train_eval = graphlearner(cell_embeds, adj_init_sparse)
            feat_train_eval = smooth_cell_features(cell_features_train, adj_train_eval, gene_feat_base)
            train_preds = []
            for input_nodes, output_nodes, blocks in train_loader:
                blocks_gpu = [b.to(args.device) for b in blocks]
                src_ids = blocks_gpu[0].srcdata[dgl.NID]
                out = model.model(blocks_gpu, feat_train_eval[src_ids])
                train_preds.append((out.argmax(1), (output_nodes - num_genes).to(args.device)))
            train_pred_full = torch.zeros(num_train_cells, dtype=torch.long, device=args.device)
            for preds, idx in train_preds:
                train_pred_full[idx] = preds
            inner_score = (train_pred_full == labels_train).float().mean()

            features_test = g_test.ndata["features"].to(args.device)
            cell_features_test = features_test[num_genes:].detach()
            gene_feat_test = features_test[:num_genes].detach()
            src_t, dst_t = g_test.edges()
            cell_mask_t = (src_t >= num_genes) & (dst_t >= num_genes)
            if cell_mask_t.any():
                weights_t = g_test.edata["weight"].squeeze(-1) if "weight" in g_test.edata \
                    else torch.ones(src_t.shape[0], device=args.device)
                adj_init_sparse_test = (src_t[cell_mask_t] - num_genes,
                                        dst_t[cell_mask_t] - num_genes,
                                        weights_t[cell_mask_t])
            else:
                adj_init_sparse_test = None

            test_embeds = cell_features_test.clone()
            for t in range(args.num_inner_iters):
                adj_test_t = graphlearner(test_embeds, adj_init_sparse_test)
                if t == args.num_inner_iters - 1:
                    feat_test_t = smooth_cell_features(cell_features_test, adj_test_t, gene_feat_test)
                else:
                    feat_test_t = smooth_cell_features(test_embeds, adj_test_t, gene_feat_test)
                    test_embeds = feat_test_t[num_genes:]

            test_preds = []
            for input_nodes, output_nodes, blocks in test_loader:
                blocks_gpu = [b.to(args.device) for b in blocks]
                src_ids = blocks_gpu[0].srcdata[dgl.NID]
                out = model.model(blocks_gpu, feat_test_t[src_ids])
                test_preds.append((out.argmax(1), (output_nodes - num_genes).to(args.device)))
            test_pred_full = torch.zeros(num_test_cells, dtype=torch.long, device=args.device)
            for preds, idx in test_preds:
                test_pred_full[idx] = preds
            score = (test_pred_full == labels_test).float().mean()

        end_time = time.time()
        run_time = end_time - start_time

        scores.append(score.item())
        inner_scores.append(inner_score.item())
        times.append(run_time)

        print(f"Run {seed} | Score: {score.item():.4f}, Time: {run_time:.2f}s")

    print(f"ScDeepSort+IDGL {args.species} {args.tissue} {args.test_dataset}:")
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_inner_score = np.mean(inner_scores)
    std_inner_score = np.std(inner_scores)

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

    # 构造文件名，例如：results_scdeepsort_mouse_Spleen_1759.json
    test_ds_str = "_".join(map(str, args.test_dataset))
    json_filename = f"results_scdeepsort_{args.species}_{args.tissue}_{test_ds_str}.json"

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    print(f"Results successfully saved to {json_filename}")