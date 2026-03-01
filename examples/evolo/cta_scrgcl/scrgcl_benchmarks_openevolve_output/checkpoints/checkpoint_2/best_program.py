import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import anndata

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.transforms import Compose, NormalizeTotalLog1P, SetConfig
from dance.utils import set_seed, sub_data
from dance.modules.single_modality.cell_type_annotation.scrgcl import scRGCLWrapper

# Start
import argparse
import os
import pandas as pd
import numpy as np
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
    - Filters reference graph by score quantile or top-k neighbors.
    - Maps Gene Symbols to Entrez IDs using MyGeneInfo.
    - Creates weighted edges based on STRING confidence scores.
    - Adds self-loops to the graph.
    - Encodes cell type labels.

    Parameters
    ----------
    net_file : str
        Path to the network CSV file (columns: node1, node2, score).
    thres : float
        Quantile threshold for network filtering (default: 0.99).
    k_neighbors : int
        Number of top neighbors to keep per node (default: None, use quantile only).
    weight_scale : str
        Method to scale edge weights ('minmax', 'softmax', or None for raw scores).
    species : str
        Species for gene mapping (e.g., 'human', 'mouse').
    out : str
        Key in `adata.uns` where the edge index will be stored.
    """

    _DISPLAY_ATTRS: Tuple[str] = ('net_file', 'thres', 'k_neighbors', 'weight_scale', 'species')

    def __init__(
        self,
        net_file=os.path.join(DANCEPKGDIR, "metadata", "STRINGDB.graph.csv"),
        thres: float = 0.99,
        k_neighbors: int = 10,
        weight_scale: str = 'minmax',
        species: str = "human",
        out: str = "edge_index",
        log_level: LogLevel = "INFO"
    ):
        super().__init__(out=out, log_level=log_level)
        self.net_file = net_file
        self.thres = thres
        self.k_neighbors = k_neighbors
        self.weight_scale = weight_scale
        self.species = species
        self.out = out
        
        # Validate threshold
        assert 0 <= thres <= 1, "quantile should be a float value in [0,1]."
        if weight_scale is not None:
            assert weight_scale in ['minmax', 'softmax'], "weight_scale must be 'minmax', 'softmax', or None"

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
        else:
            return scores

    def _apply_top_k_filter(self, df: pd.DataFrame, k: int) -> pd.DataFrame:
        """Apply top-k filtering per node to create a more balanced graph."""
        if k is None:
            return df
        
        # Create both directions for undirected graph
        df_bidir = pd.concat([
            df[['node1', 'node2', 'score']],
            df.rename(columns={'node1': 'node2', 'node2': 'node1'})[['node1', 'node2', 'score']]
        ], ignore_index=True)
        
        # For each node, keep top-k highest scoring connections
        result_edges = []
        for node in df_bidir['node1'].unique():
            node_edges = df_bidir[df_bidir['node1'] == node].nlargest(k, 'score')
            result_edges.append(node_edges)
        
        if result_edges:
            result_df = pd.concat(result_edges, ignore_index=True)
            # Remove duplicates (bidirectional edges)
            result_df = result_df.drop_duplicates(subset=['node1', 'node2'], keep='first')
            return result_df
        else:
            return df

    def _add_remaining_self_loop(self, edge_df: pd.DataFrame, num_nodes: int, fill_value: float = 1.0) -> pd.DataFrame:
        """Adds self-loops (node_i, node_i) to nodes that don't have them."""
        assert 'node1' in edge_df.columns
        assert 'node2' in edge_df.columns
        
        edge_index = edge_df[['node1', 'node2']].T.values
        row, col = edge_index[0], edge_index[1]
        
        N = num_nodes if num_nodes is not None else np.max(edge_index) + 1
        
        mask = row == col
        existing_loops = set(row[mask])
        all_nodes = set(np.arange(0, N, dtype=int))
        added_index = list(all_nodes - existing_loops)
        
        if not added_index:
            return edge_df
            
        new_df = pd.DataFrame()
        new_df['node1'] = added_index
        new_df['node2'] = added_index
        new_df['score'] = fill_value  # Use score column for self-loops
            
        edge_df = pd.concat([edge_df, new_df], ignore_index=True)
        return edge_df

    def _map_graph_to_genes(self, graph_df: pd.DataFrame, gene_list: List[str]) -> tuple:
        """
        Maps graph edges (Symbols/Entrez) to the indices of the provided gene_list.
        Returns both edge index and edge weights.
        """
        # 1. Prepare Graph
        graph_edge_df = graph_df.copy()
        graph_edge_df.columns = ['node1', 'node2', 'score']
        graph_edge_df = graph_edge_df.astype({'node1': str, 'node2': str, 'score': float})
        
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
        
        # 3. Build Mapping (Symbol -> Index in gene_list)
        symbol_to_index_dict = {}
        for item in res:
            if 'entrezgene' in item and 'query' in item:
                original_symbol = item['query']
                
                if original_symbol in symbol_to_idx_dict:
                    idx = symbol_to_idx_dict[original_symbol]
                    symbol_to_index_dict[original_symbol] = idx
            # Also try direct symbol mapping as fallback
            elif 'query' in item and item['query'] in symbol_to_idx_dict:
                original_symbol = item['query']
                idx = symbol_to_idx_dict[original_symbol]
                symbol_to_index_dict[original_symbol] = idx

        self.logger.info(f"Mapping complete: {len(symbol_to_idx_dict)} symbols mapped to {len(symbol_to_index_dict)} indices.")
        
        # 4. Apply Mapping
        graph_edge_df['node1'] = graph_edge_df['node1'].map(symbol_to_index_dict)
        graph_edge_df['node2'] = graph_edge_df['node2'].map(symbol_to_index_dict)
        
        # Remove edges where nodes weren't found in the gene list
        graph_edge_df = graph_edge_df.dropna().astype({'node1': int, 'node2': int, 'score': float})
        
        # 5. Apply top-k filtering if specified
        if self.k_neighbors is not None and self.k_neighbors > 0:
            graph_edge_df = self._apply_top_k_filter(graph_edge_df, self.k_neighbors)
        
        # 6. Scale weights if specified
        if self.weight_scale is not None and len(graph_edge_df) > 0:
            graph_edge_df['score'] = self._scale_weights(graph_edge_df['score'].values, self.weight_scale)
        
        # 7. Add Self Loops
        graph_edge_df = self._add_remaining_self_loop(
            graph_edge_df, 
            num_nodes=len(gene_list),
            fill_value=1.0
        )
        
        # Return both edge index and weights
        edge_index = graph_edge_df[['node1', 'node2']].values.T
        edge_weights = graph_edge_df['score'].values if 'score' in graph_edge_df.columns else np.ones(len(graph_edge_df))
        
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
        
        self.logger.info(f"Graph filtered. Retained {len(filtered_graph)} edges with score >= {cutoff_value:.4f}")

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
        self.logger.info(f'Edge weights shape: {edge_weights.shape}')

        # 5. Save results to adata.uns
        adata.uns[self.out] = edge_index
        adata.uns[f'{self.out}_weights'] = edge_weights  # Store edge weights
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
    for run in range(runs):
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
            print(f"Run {run+1} Score: {score:.4f}")

    print(f"\nscRGCL {args.species} {args.tissue} Test Set {args.test_dataset}:")
    print(f"mean_score: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
    print(f"mean_inner_score: {np.mean(inner_scores):.5f} +/- {np.std(inner_scores):.5f}")