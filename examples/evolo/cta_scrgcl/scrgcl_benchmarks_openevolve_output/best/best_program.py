import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import anndata
import time  # 新增：导入 time 模块

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.transforms import Compose, NormalizeTotalLog1P, SetConfig
from dance.utils import set_seed, sub_data
from dance.modules.single_modality.cell_type_annotation.scrgcl import scRGCLWrapper

# Start
import pandas as pd
import anndata as ad
import mygene
import requests_cache
import hashlib
import logging
import tempfile
from abc import ABC, abstractmethod

from dance.data import Data
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import List, LogLevel, Tuple
from dance.settings import DANCEPKGDIR
    
# EVOLVE-BLOCK-START
@register_preprocessor("graph", "cell",overwrite=True)
class StringDBGraph(BaseTransform):
    """
    Generates a gene-gene interaction graph for an AnnData object by filtering 
    a reference network (e.g., STRING) and mapping it to the gene expression features.

    Features:
    - Filters reference graph by score quantile.
    - Maps Gene Symbols to Entrez IDs using MyGeneInfo.
    - Adds self-loops to the graph.
    - Supports weighted edges and top-k neighbor selection.
    - Encodes cell type labels.

    Parameters
    ----------
    net_file : str
        Path to the network CSV file (columns: node1, node2, score).
    thres : float
        Quantile threshold for network filtering (default: 0.99).
    species : str
        Species for gene mapping (e.g., 'human', 'mouse').
    out : str
        Key in `adata.uns` where the edge index will be stored.
    k_neighbors : int
        Number of top neighbors to keep per gene (default: 10).
    weight_scale : str
        Method to scale edge weights: 'minmax', 'softmax', or 'none' (default: 'minmax').
    """

    _DISPLAY_ATTRS: Tuple[str] = ('net_file', 'thres', 'species', 'k_neighbors', 'weight_scale')

    def __init__(
        self,
        net_file=os.path.join(DANCEPKGDIR, "metadata", "STRINGDB.graph.csv"),
        thres: float = 0.99,
        species: str = "human",
        out: str = "edge_index",
        k_neighbors: int = 10,
        weight_scale: str = "minmax",
        log_level: LogLevel = "INFO"
    ):
        super().__init__(out=out, log_level=log_level)
        self.net_file = net_file
        self.thres = thres
        self.species = species
        self.k_neighbors = k_neighbors
        self.weight_scale = weight_scale
        self.out = out
        
        # Validate threshold
        assert 0 <= thres <= 1, "quantile should be a float value in [0,1]."
        assert weight_scale in ['minmax', 'softmax', 'none'], "weight_scale must be 'minmax', 'softmax', or 'none'"

    def _scale_weights(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Scale edge weights using specified method."""
        if method == 'minmax':
            min_score, max_score = scores.min(), scores.max()
            if max_score == min_score:
                return np.ones_like(scores)
            return (scores - min_score) / (max_score - min_score)
        elif method == 'softmax':
            exp_scores = np.exp(scores - scores.max())  # Subtract max for numerical stability
            return exp_scores / exp_scores.sum()
        else:  # 'none'
            return scores

    def _top_k_prune(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Keep top-k highest scoring neighbors for each node."""
        # Create bidirectional mapping to handle both directions
        df_bidir = pd.concat([
            df[['node1', 'node2', 'score']],
            df[['node2', 'node1', 'score']].rename(columns={'node2': 'node1', 'node1': 'node2'})
        ], ignore_index=True)
        
        # For each node, keep only top-k neighbors
        result_edges = []
        for node in df_bidir['node1'].unique():
            node_edges = df_bidir[df_bidir['node1'] == node].nlargest(k, 'score')
            result_edges.append(node_edges)
        
        if result_edges:
            pruned_df = pd.concat(result_edges, ignore_index=True)
            # Remove duplicates (since we created bidirectional edges)
            pruned_df = pruned_df.drop_duplicates(subset=['node1', 'node2'])
            return pruned_df
        else:
            return df

    def _add_remaining_self_loop(self, edge_df: pd.DataFrame, num_nodes: int, fill_value: float = 1.0) -> pd.DataFrame:
        """Adds self-loops (node_i, node_i) to nodes that don't have them."""
        assert 'node1' in edge_df.columns
        assert 'node2' in edge_df.columns
        
        # Get unique nodes from the current edges
        if len(edge_df) > 0:
            current_nodes = set(edge_df['node1'].unique()) | set(edge_df['node2'].unique())
        else:
            current_nodes = set()
        
        N = num_nodes if num_nodes is not None else max(current_nodes) + 1 if current_nodes else 0
        
        # Identify nodes that don't have self-loops
        all_nodes = set(range(N))
        nodes_without_loops = all_nodes - current_nodes
        
        # Find existing self-loops in the dataframe
        existing_self_loops = set(edge_df[edge_df['node1'] == edge_df['node2']]['node1'].values)
        nodes_without_loops = nodes_without_loops | (all_nodes - existing_self_loops)
        
        if not nodes_without_loops:
            return edge_df
            
        # Add self-loops for nodes that don't have them
        new_df = pd.DataFrame()
        new_df['node1'] = list(nodes_without_loops)
        new_df['node2'] = list(nodes_without_loops)
        new_df['score'] = fill_value  # Add score column for consistency
            
        edge_df = pd.concat([edge_df, new_df], ignore_index=True)
        return edge_df

    def _map_graph_to_genes(self, graph_df: pd.DataFrame, gene_list: List[str]) -> tuple:
        """
        Maps graph edges (Symbols/Entrez) to the indices of the provided gene_list.
        Returns both edge_index and edge_weights.
        """
        # 1. Prepare Graph
        graph_edge_df = graph_df.copy()
        graph_edge_df.columns = ['node1', 'node2', 'score']
        graph_edge_df = graph_edge_df.astype(str)
        
        # 2. Prepare MyGene Query
        symbol_to_idx_dict = {g.strip(): idx for idx, g in enumerate(gene_list)}
        
        requests_cache.install_cache(
            'mygene_cache', 
            expire_after=3600*24*30, 
            allowable_methods=['GET', 'POST']
        )
        mg = mygene.MyGeneInfo()
        
        self.logger.info(f"Querying MyGene info for {len(symbol_to_idx_dict)} genes...")
        res = mg.querymany(
            symbol_to_idx_dict.keys(), 
            scopes='symbol,alias', 
            fields='entrezgene', 
            species=self.species, 
            verbose=False
        )
        
        # 3. Build Mapping (Symbol -> Index in gene_list) - use original symbols for mapping
        symbol_to_index_dict = {}
        for item in res:
            if 'entrezgene' in item and 'query' in item:
                original_symbol = item['query']
                
                if original_symbol in symbol_to_idx_dict:
                    idx = symbol_to_idx_dict[original_symbol]
                    symbol_to_index_dict[original_symbol] = idx
            # Also try mapping directly if the query is already a symbol in our list
            elif 'query' in item and item['query'] in symbol_to_idx_dict:
                original_symbol = item['query']
                idx = symbol_to_idx_dict[original_symbol]
                symbol_to_index_dict[original_symbol] = idx

        self.logger.info(f"Mapping complete: {len(symbol_to_idx_dict)} symbols mapped to {len(symbol_to_index_dict)} indices.")

        # 4. Apply Mapping - first convert scores back to numeric for processing
        graph_edge_df['score'] = pd.to_numeric(graph_edge_df['score'], errors='coerce')
        graph_edge_df = graph_edge_df.dropna(subset=['score'])  # Remove rows with invalid scores
        
        # Map node names to indices
        graph_edge_df['node1'] = graph_edge_df['node1'].map(symbol_to_index_dict)
        graph_edge_df['node2'] = graph_edge_df['node2'].map(symbol_to_index_dict)
        
        # Remove edges where nodes weren't found in the gene list
        graph_edge_df = graph_edge_df.dropna().astype({'node1': int, 'node2': int, 'score': float})
        
        # 5. Apply top-k pruning if specified
        if self.k_neighbors > 0 and len(graph_edge_df) > 0:
            graph_edge_df = self._top_k_prune(graph_edge_df, self.k_neighbors)
        
        # 6. Scale weights if specified
        if len(graph_edge_df) > 0:
            scores = graph_edge_df['score'].values
            scaled_scores = self._scale_weights(scores, self.weight_scale)
            graph_edge_df['score'] = scaled_scores
        
        # 7. Add Self Loops
        graph_edge_df = self._add_remaining_self_loop(
            graph_edge_df, 
            num_nodes=len(gene_list),
            fill_value=graph_edge_df['score'].mean() if len(graph_edge_df) > 0 else 1.0
        )
        
        # Separate edge index and weights
        if len(graph_edge_df) > 0:
            edge_index = graph_edge_df[['node1', 'node2']].values.T
            edge_weights = graph_edge_df['score'].values
        else:
            # If no edges remain after filtering, create minimal graph with self-loops
            n_genes = len(gene_list)
            edge_index = np.array([[i for i in range(n_genes)], [i for i in range(n_genes)]])
            edge_weights = np.ones(n_genes)
        
        return edge_index, edge_weights

    def __call__(self, data: Data) -> Data:
        """
        Process the AnnData object: load graph, filter, map, and update adata.uns.
        """
        self.logger.info(f"Processing data with threshold {self.thres}, k_neighbors={self.k_neighbors}, weight_scale={self.weight_scale}")
        adata = data.data
        # 1. Load and Filter Network
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"Network file not found: {self.net_file}")

        graph_df = pd.read_csv(self.net_file, header=None, index_col=None)
        # Ensure 3 columns: node1, node2, score
        if graph_df.shape[1] < 3:
             # Handle case if CSV doesn't have scores, assumes weight 1 or needs handling
             raise ValueError("Network file must have at least 3 columns: node1, node2, score")
             
        graph_df = graph_df.iloc[:, :3]
        graph_df.columns = ['node1', 'node2', 'score']
        
        # Quantile filtering
        cutoff_value = graph_df['score'].quantile(self.thres)
        filtered_graph = graph_df.loc[graph_df['score'].ge(cutoff_value)]
        
        self.logger.info(f"Graph filtered. Retained edges with score >= {cutoff_value:.4f} ({len(filtered_graph)} edges)")

        # 2. Extract Gene Names and Process Graph
        # Check for gene names in .var_names or .var.index
        gene_names = adata.var_names.tolist()
        
        edge_index, edge_weights = self._map_graph_to_genes(filtered_graph, gene_names)
        
        # 3. Process Cell Labels (if 'cell_type' exists)
        if 'cell_type' in adata.obs:
            label_df = adata.obs['cell_type']
            str_labels = np.unique(label_df.values).tolist()
            # Convert string labels to integer indices
            label_indices = [str_labels.index(x) for x in label_df.values]
            
            # Update obs to ensure we have the processed version (optional, mimics gen_data)
            # Note: The original gen_data created a NEW adata. 
            # Here we attach info to the existing one to match the transform pattern.
            adata.uns['str_labels'] = str_labels
            # We assume the user might want these accessible easily
            self.logger.info(f"Cell types found: {len(str_labels)}")
        else:
            self.logger.warning("'cell_type' column not found in adata.obs. Skipping label processing.")
            str_labels = []

        # 4. Log statistics
        self.logger.info(f'Shape of expression matrix: {adata.shape}')
        self.logger.info(f'Shape of backbone network: {edge_index.shape}')
        self.logger.info(f'Edge weights range: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]')

        # 5. Save results to adata.uns
        adata.uns[self.out] = edge_index
        adata.uns[f'{self.out}_weight'] = edge_weights  # Store edge weights for GNN
        adata.uns['thres'] = self.thres
        
        # If strict compatibility with `gen_data` output structure is required:
        # The original function returned a NEW adata. 
        # Here we modify in place. If you strictly need the 'cleaned' structure 
        # (just X, cell_type, var), you would do it here, but typically 
        # keeping the full object is better.
        
        self.logger.info('Finished processing data.')
        return data
# EVOLVE-BLOCK-END


def get_preprocessing_pipeline(log_level="INFO",thres= 0.99, species= "human"):
        return Compose(
            NormalizeTotalLog1P(),
            StringDBGraph(thres= thres, species= species),
            SetConfig({
                "label_channel": "cell_type"
            }),
            log_level=log_level,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[138], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Brain", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[328], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    parser.add_argument("--species", default="human", type=str)
    parser.add_argument("--cache", action="store_true", help="Cache processed data") 
    
    # scRGCL Specific parameters
    parser.add_argument('--quantile', type=float, default=0.99,
                        help='Quantile threshold for network filtering')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory for models')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model parameters
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument("--obs_nums",type=int,default=None)
    args = parser.parse_args()
    runs = args.num_runs
    
    # Check GPU availability
    device = torch.device("cuda:" + str(args.gpu))
    print(f"Running on {device}")

    scores = []
    inner_scores=[]
    times = []  # 新增：用于记录每次运行的时间
    
    for run in range(runs):
        start_time = time.time()  # 新增：记录单次循环的开始时间
        
        set_seed(args.seed + run)
        with tempfile.TemporaryDirectory() as temp_dir:
            model = scRGCLWrapper(
                out_dir=temp_dir,
                dropout_ratio=args.dropout_ratio,
                init_lr=args.init_lr,
                seed=args.seed + run,
                device=device,
            )
            
            preprocessing_pipeline = get_preprocessing_pipeline(thres=args.quantile, species=args.species)
            # 1. Load Data using DANCE
            dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
            data = dataloader.load_data(transform=None, cache=args.cache)
            if args.obs_nums is not None:
                sub_data(data.data,args.obs_nums)
                train_idx, test_idx = train_test_split(range(args.obs_nums),test_size=0.2,random_state=args.seed + run)
                train_idx,val_idx = train_test_split(train_idx,test_size=args.val_size,random_state=args.seed + run)
                data.set_split_idx("train", train_idx)
                data.set_split_idx("test", test_idx)
                data.set_split_idx("val", val_idx)
            preprocessing_pipeline(data)
            # 2. Extract Train/Test Data
            # return_type="torch" ensures we get tensors, but we might need numpy for AnnData creation
            x_train, y_train = data.get_train_data(return_type="torch")
            x_val,y_val=data.get_val_data(return_type="torch")
            x_test, y_test = data.get_test_data(return_type="torch")
            
            # 3. Convert Labels (One-hot -> Index)
            # scRGCLWrapper internally expects 1D array of indices or strings
            y_train_indices = y_train.argmax(1).cpu().numpy()
            
            # 4. Construct AnnData for Training
            # The scRGCL wrapper fit method expects an AnnData object to run gen_data internally
            train_adata = anndata.AnnData(X=x_train.cpu().numpy())
            train_adata.uns=data.data.uns
            train_adata.obs['cell_type'] = y_train_indices
            
            # If gen_data relies on gene names (feature names), we should attach them.
            # DANCE data object usually holds feature info.
            if hasattr(data.data, "var_names"):
                train_adata.var_names = data.data.var_names

            # 5. Initialize Model
            # Note: device handling is largely done inside fit(), but we pass params here

            # 6. Fit the Model
            # Pass the constructed adata and specific args
            print(f"--- Run {run+1}/{runs}: Fitting model ---")
            model.fit(adata=train_adata, batch_size=args.batch_size)
            # 7. Evaluate
            # BaseMethod.score() will call self.predict(x_test) -> score_func(y_test, y_pred)
            # We pass x_test (Tensor or Numpy) which the wrapper handles in _prepare_test_loader
            print(f"--- Run {run+1}/{runs}: Evaluating ---")
            score = model.score(x_test, y_test, score_func="acc")
            inner_score = model.score(x_val, y_val, score_func="acc")
            scores.append(score)
            inner_scores.append(inner_score)
            
            end_time = time.time()  # 新增：记录单次循环的结束时间
            run_time = end_time - start_time
            times.append(run_time)  # 新增：保存耗时
            
            print(f"Run {run+1} Score: {score:.4f}, inner_score: {inner_score:.4f}, time: {run_time:.2f}s")  # 修改：加入内部分数和运行时间打印

    print(f"\nscRGCL {args.species} {args.tissue} Test Set {args.test_dataset}:")
    # 修改：加入 times 列表，以供后续捕获
    print(f"scores:{scores},inner_scores:{inner_scores},times:{times}")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")
    print(f"mean_time: {np.mean(times):.2f}s")  # 新增：输出平均运行时间