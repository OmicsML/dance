import argparse
import time  # 新增：导入 time 模块
from typing import Optional

import dgl
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
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



# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell",overwrite=True)
class HeteronetGraph(BaseTransform):

    def __init__(self, knn_num: int = 5, distance_metrics: str = 'l2', random_state: int = 0,
                 mutual: bool = True, add_self_loop: bool = True, threshold: Optional[float] = None,
                 channel: Optional[str] = None, channel_type: Optional[str] = "X", ignore_first: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.knn_num = knn_num
        self.distance_metrics = distance_metrics
        self.random_state = random_state
        self.mutual = mutual
        self.add_self_loop = add_self_loop
        self.threshold = threshold  # Distance threshold for edge pruning
        self.channel = channel
        self.ignore_first = ignore_first
        self.channel_type = channel_type

    def _compute_mutual_knn_indices(self, features_tensor, k, threshold=None):
        """Computes mutual KNN indices for more robust graph construction."""
        # Use DGL's built-in KNN function for efficiency
        g_knn = dgl.knn_graph(features_tensor, k, algorithm='bruteforce-blas')
        src, dst = g_knn.edges()

        if self.mutual:
            # Efficient mutual KNN computation using tensor operations
            num_nodes = features_tensor.shape[0]
            
            # Create CSR-like sparse matrix for neighbor lookups
            row_indices = src
            col_indices = dst
            
            # For each edge (u,v), check if (v,u) also exists
            edge_pairs = torch.stack([src, dst], dim=1)
            reversed_edge_pairs = torch.stack([dst, src], dim=1)
            
            # Sort both edge sets to efficiently find mutual neighbors
            sorted_indices = torch.argsort(row_indices)
            sorted_row_indices = row_indices[sorted_indices]
            sorted_col_indices = col_indices[sorted_indices]
            
            # Create a map of each node to its neighbors for mutual checking
            node_to_neighbors = {}
            for i in range(len(sorted_row_indices)):
                u = sorted_row_indices[i].item()
                v = sorted_col_indices[i].item()
                if u not in node_to_neighbors:
                    node_to_neighbors[u] = set()
                node_to_neighbors[u].add(v)
            
            # Find mutual neighbors
            mutual_src_list = []
            mutual_dst_list = []
            
            for u in node_to_neighbors:
                neighbors = node_to_neighbors[u]
                for v in neighbors:
                    # Check if u is also in v's neighbors (mutual)
                    if u in node_to_neighbors.get(v, set()):
                        mutual_src_list.append(u)
                        mutual_dst_list.append(v)
            
            if mutual_src_list:
                src = torch.tensor(mutual_src_list, dtype=torch.long)
                dst = torch.tensor(mutual_dst_list, dtype=torch.long)
        
        # Apply distance threshold if provided
        if threshold is not None and len(src) > 0:
            # Calculate distances for current edges
            coords_selected = features_tensor[src]
            neighbor_coords = features_tensor[dst]
            distances = torch.sqrt(torch.sum((coords_selected - neighbor_coords) ** 2, dim=1))
            
            # Keep only edges within threshold
            mask = distances <= threshold
            src = src[mask]
            dst = dst[mask]
        
        return src, dst

    def __call__(self, data):
        """Builds a DGL graph from an AnnData object.

        Args:
            adata: The AnnData object containing features, labels, and splits.
            ref_adata_name: Name for the dataset (used if needed later).
            knn_num: Number of nearest neighbors for graph construction.
            distance_metrics: Distance metric for KNN.
            mutual: Whether to use mutual KNN for more robust connections.
            add_self_loop: Whether to add self loops to preserve original features.
            threshold: Distance threshold to prune noisy edges.
            ignore_first: If True, sets label 0 to -1.

        Returns:
            dgl.DGLGraph: A DGL graph with node features ('feat'), labels ('label'),
                        and split masks ('train_mask', 'val_mask', 'test_mask',
                        'id_mask', 'ood_mask').

        """
        adata = data.data
        # 1. Extract Features
        features_np = data.get_feature(return_type="numpy", channel=self.channel, channel_type=self.channel_type)
        
        # Apply biologically informed convex combination of gene embeddings if available
        # Based on cta_scdeepsort reference: fuse direct cell-level TruncatedSVD embeddings 
        # with weighted gene-PCA embeddings using biologically informed convex combination (0.6/0.4)
        if hasattr(adata, 'obsm') and 'X_pca' in adata.obsm.keys():
            pca_features = adata.obsm['X_pca']
            # Convex combination of original features and PCA features
            combined_features = 0.6 * features_np + 0.4 * pca_features
            features = torch.as_tensor(combined_features, dtype=torch.float32)
        else:
            features = torch.as_tensor(features_np, dtype=torch.float32)  # Ensure float32 common for features
        
        num_nodes = features.shape[0]

        # 2. Extract Labels
        # Assuming labels are one-hot encoded in obsm['cell_type']
        labels_np = np.argmax(adata.obsm['cell_type'].copy(), axis=1)
        labels = torch.as_tensor(labels_np, dtype=torch.long)

        batchs = adata.obs.get('batch_id', None)

        if self.ignore_first:
            labels[labels == 0] = -1  # Apply ignore_first logic

        # 3. Build Edges using mutual KNN for more robust graph
        src, dst = self._compute_mutual_knn_indices(features, self.knn_num, self.threshold)

        # Handle case where no edges exist after mutual KNN and threshold
        if len(src) == 0:
            # Create minimal graph with self loops only
            src = torch.arange(num_nodes, dtype=torch.long)
            dst = torch.arange(num_nodes, dtype=torch.long)

        # 4. Create DGL Graph
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        
        # 5. Add self-loops if specified
        if self.add_self_loop:
            g = dgl.add_self_loop(g)

        # 6. Add Node Features and Labels
        g.ndata['feat'] = features
        g.ndata['label'] = labels
        if batchs is not None:
            g.ndata['batch_id'] = torch.from_numpy(batchs.values.astype(int)).long()
        adata.uns[self.out] = g
# EVOLVE-BLOCK-END

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
    parser.add_argument("--obs_nums",type=int,default=None)
    args = parser.parse_args()
    runs = args.num_runs
    results = []
    inner_scores = []
    times = []  # 新增：用于记录每次运行的时间
    
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    eval_func = eval_acc
    
    for run in range(runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间
        
        set_seed(args.seed + run)
        dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                               train_dataset=args.train_dataset, data_dir=args.data_dir,
                                               val_size=args.val_size)
        ref_data_name = f"{args.species}_{args.tissue}_{args.train_dataset}"
        preprocessing_pipeline = get_preprocessing_pipeline()
        data = dataloader.load_data(transform=None, cache=args.cache)
        if args.obs_nums is not None:
            sub_data(data.data,args.obs_nums)
            train_idx, test_idx = train_test_split(range(args.obs_nums),test_size=0.2,random_state=args.seed)
            train_idx,val_idx = train_test_split(train_idx,test_size=args.val_size,random_state=args.seed)
            data.set_split_idx("train", train_idx)
            data.set_split_idx("test", test_idx)
            data.set_split_idx("val", val_idx)
        preprocessing_pipeline(data)
        set_split(data, data.train_idx, data.val_idx, data.test_idx)
        # dataset_ind, dataset_ood_tr, dataset_ood_te, adata = load_cell_graph_fixed(
        #     data.data, ref_data_name)
        g = data.data.uns['HeteronetGraph']
        # set_graph_split(data.data,ref_data_name,g)
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
        for epoch in range(args.epochs):
            loss = model.fit(dataset_ind, dataset_ood_tr, args.use_zinb, adata, args.zinb_weight, args.cl_weight,
                             args.mask_ratio, criterion, optimizer)
            # model.evaluate(dataset_ind, dataset_ood_te, criterion, eval_func, args.display_step, run, results, epoch,
            #                loss, f"{args.species}_{args.tissue}_{args.train_dataset}", args.T, args.use_prop,
            #                args.use_2hop, args.oodprop, args.oodalpha)
            # test_idx=dataset_ind.splits['test']
            test_score = model.score(dataset_ind, dataset_ind.y, data.test_idx)
            inner_score = model.score(dataset_ind, dataset_ind.y, data.train_idx)
            
        results.append(test_score)
        inner_scores.append(inner_score)
        
        end_time = time.time()  # 新增：记录单次循环的结束时间
        run_time = end_time - start_time
        times.append(run_time)  # 新增：保存耗时
        
        print(f"Run {run+1} - test_score: {test_score:.4f}, time: {run_time:.2f}s")  # 新增：打印单次运行时间和得分

    print(f"scHeteroNet {args.species} {args.tissue} {args.test_dataset}:")
    # 修改：将 results 作为 scores 列表打印，加入 times，以供 evaluator 捕获
    print(f"scores:{results},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(results):.5f} +/- {np.std(results):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间

#TODO test_score is true delete odd test以及其他评估方法，再次测试，然后将valid等和其他算法保持一致。
"""

Epoch: 00, Loss: 2.3264, AUROC: 44.03%, AUPR: 99.12%, FPR95: 100.00%, Test Score: 46.85%
Run 01:
Chosen epoch: 4
OOD Test 1 Final AUROC: 64.61
OOD Test 1 Final AUPR: 99.53
OOD Test 1 Final FPR95: 100.00
IND Test Score: 81.88
All runs:
OOD Test 1 Final AUROC: 64.61
OOD Test 1 Final AUPR: 99.53
OOD Test 1 Final FPR: 100.00
IND Test Score: 81.88

python scheteronet.py --gpu -1 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue Brain --train_dataset 328 --test_dataset 138 --use_zinb --use_prop --use_2hop
python scheteronet.py --gpu -1  --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --use_zinb --use_prop --use_2hop
python scheteronet.py --gpu 0  --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu 0  --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 1729 2125 2184 2724 2743 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu 0  --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 1925 205 3323 6509 7572 --use_zinb --use_prop --use_2hop
"""