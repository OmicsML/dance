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
    - Maps Gene Symbols to Entrez IDs using MyGeneInfo with fallback.
    - Creates weighted edges based on STRING scores.
    - Applies top-k pruning to prevent hub formation.
    - Adds self-loops to the graph.
    - Encodes cell type labels.

    Parameters
    ----------
    net_file : str
        Path to the network CSV file (columns: node1, node2, score).
    thres : float
        Quantile threshold for initial network filtering (default: 0.95).
    species : str
        Species for gene mapping (e.g., 'human', 'mouse').
    out : str
        Key in `adata.uns` where the edge index will be stored.
    k_neighbors : int
        Number of top neighbors to keep per gene after filtering (default: 10).
    weight_scale : str
        Method to scale edge weights ('minmax', 'softmax', 'none') (default: 'minmax').
    """

    _DISPLAY_ATTRS: Tuple[str] = ('net_file', 'thres', 'species', 'k_neighbors', 'weight_scale')

    def __init__(
        self,
        net_file=os.path.join(DANCEPKGDIR, "metadata", "STRINGDB.graph.csv"),
        thres: float = 0.95,
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
        self.out = out
        self.k_neighbors = k_neighbors
        self.weight_scale = weight_scale
        
        # Validate threshold
        assert 0 <= thres <= 1, "quantile should be a float value in [0,1]."
        assert weight_scale in ['minmax', 'softmax', 'none'], "weight_scale must be 'minmax', 'softmax', or 'none'"

    def _scale_weights(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Scale edge weights using specified method."""
        if method == 'minmax':
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                return (scores - min_score) / (max_score - min_score)
            else:
                return np.ones_like(scores)
        elif method == 'softmax':
            exp_scores = np.exp(scores - scores.max())  # Subtract max for numerical stability
            return exp_scores / exp_scores.sum()
        else:  # none
            return scores

    def _top_k_prune(self, edge_df: pd.DataFrame, k: int, gene_count: int) -> pd.DataFrame:
        """Keep only top-k strongest connections per gene."""
        if len(edge_df) == 0:
            return edge_df
            
        # Create bidirectional mapping to handle both directions
        # For each gene, keep top-k highest scoring connections
        result_edges = []
        
        # Get unique genes involved in edges
        all_genes = set()
        all_genes.update(edge_df['node1'].unique())
        all_genes.update(edge_df['node2'].unique())
        
        # For each gene, find top-k connections
        for gene in all_genes:
            # Find all edges involving this gene (both as source and target)
            gene_mask = (edge_df['node1'] == gene) | (edge_df['node2'] == gene)
            gene_edges = edge_df[gene_mask].copy()
            
            if len(gene_edges) <= k:
                result_edges.append(gene_edges)
            else:
                # Sort by score descending and take top-k
                gene_edges_sorted = gene_edges.sort_values('score', ascending=False).head(k)
                result_edges.append(gene_edges_sorted)
        
        if result_edges:
            pruned_df = pd.concat(result_edges, ignore_index=True)
            # Remove duplicates that might have been introduced by bidirectional selection
            pruned_df = pruned_df.drop_duplicates(subset=['node1', 'node2'], keep='first')
            return pruned_df
        else:
            return edge_df

    def _add_remaining_self_loop(self, edge_df: pd.DataFrame, num_nodes: int, fill_value: float = 1.0) -> pd.DataFrame:
        """Adds self-loops (node_i, node_i) to nodes that don't have them."""
        if len(edge_df) == 0:
            # If no edges exist, create self-loops for all nodes
            new_df = pd.DataFrame()
            new_df['node1'] = range(num_nodes)
            new_df['node2'] = range(num_nodes)
            new_df['score'] = fill_value
            return new_df
            
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
        new_df['score'] = fill_value
            
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
        
        # 3. Build Mapping (Symbol -> Index in gene_list) with fallback
        symbol_to_index_dict = {}
        for item in res:
            original_symbol = item['query']
            if 'entrezgene' in item:
                # Use entrez ID if available
                entrez_id = str(item['entrezgene'])
                if original_symbol in symbol_to_idx_dict:
                    idx = symbol_to_idx_dict[original_symbol]
                    symbol_to_index_dict[entrez_id] = idx
            else:
                # Fallback: try using the symbol directly if entrez not found
                if original_symbol in symbol_to_idx_dict:
                    idx = symbol_to_idx_dict[original_symbol]
                    symbol_to_index_dict[original_symbol] = idx

        self.logger.info(f"Mapping complete: {len(symbol_to_idx_dict)} symbols mapped to {len(symbol_to_index_dict)} IDs.")

        # 4. Apply Mapping - first try with entrez IDs, then fall back to symbols
        graph_edge_df['node1_mapped'] = graph_edge_df['node1'].map(symbol_to_index_dict)
        graph_edge_df['node2_mapped'] = graph_edge_df['node2'].map(symbol_to_index_dict)
        
        # Remove edges where nodes weren't found in the gene list
        graph_edge_df = graph_edge_df.dropna()
        
        if len(graph_edge_df) == 0:
            self.logger.warning("No edges could be mapped to the gene list. Creating self-loops only.")
            # Return empty graph with self-loops for all genes
            empty_df = pd.DataFrame(columns=['node1', 'node2', 'score'])
            return empty_df.values, np.array([])

        graph_edge_df = graph_edge_df.astype({'node1_mapped': int, 'node2_mapped': int})
        
        # Create final edge dataframe with mapped indices
        final_edge_df = pd.DataFrame()
        final_edge_df['node1'] = graph_edge_df['node1_mapped'].values
        final_edge_df['node2'] = graph_edge_df['node2_mapped'].values
        final_edge_df['score'] = graph_edge_df['score'].values
        
        return final_edge_df.values, final_edge_df['score'].values

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
             raise ValueError("Network file must have at least 3 columns: node1, node2, score")
             
        graph_df = graph_df.iloc[:, :3]
        graph_df.columns = ['node1', 'node2', 'score']
        
        # Quantile filtering
        cutoff_value = graph_df['score'].quantile(self.thres)
        filtered_graph = graph_df.loc[graph_df['score'] >= cutoff_value].copy()
        
        self.logger.info(f"Graph filtered. Retained {len(filtered_graph)} edges with score >= {cutoff_value:.4f}")

        # 2. Extract Gene Names and Process Graph
        gene_names = adata.var_names.tolist()
        
        # Map graph to gene indices and get weights
        edge_data, original_weights = self._map_graph_to_genes(filtered_graph, gene_names)
        
        if len(edge_data) == 0:
            # If no edges were mapped, create self-loops for all genes
            n_genes = len(gene_names)
            edge_index = np.stack([np.arange(n_genes), np.arange(n_genes)])
            edge_weights = np.ones(n_genes)
        else:
            # Process the mapped edges
            mapped_df = pd.DataFrame(edge_data, columns=['node1', 'node2', 'score'])
            
            # Apply top-k pruning to prevent hub formation
            if self.k_neighbors > 0:
                mapped_df = self._top_k_prune(mapped_df, self.k_neighbors, len(gene_names))
                self.logger.info(f"After top-{self.k_neighbors} pruning: {len(mapped_df)} edges remain")
            
            # Add self-loops to ensure all genes are connected
            mapped_df = self._add_remaining_self_loop(
                mapped_df, 
                num_nodes=len(gene_names),
                fill_value=mapped_df['score'].median() if len(mapped_df) > 0 else 1.0
            )
            
            # Extract edge index and weights
            edge_index = mapped_df[['node1', 'node2']].T.values
            weights = mapped_df['score'].values
            
            # Scale weights according to specified method
            if len(weights) > 0:
                edge_weights = self._scale_weights(weights, self.weight_scale)
            else:
                edge_weights = np.array([])
        
        # 3. Process Cell Labels (if 'cell_type' exists)
        if 'cell_type' in adata.obs:
            label_df = adata.obs['cell_type']
            str_labels = np.unique(label_df.values).tolist()
            label_indices = [str_labels.index(x) for x in label_df.values]
            
            adata.uns['str_labels'] = str_labels
            self.logger.info(f"Cell types found: {len(str_labels)}")
        else:
            self.logger.warning("'cell_type' column not found in adata.obs. Skipping label processing.")
            str_labels = []

        # 4. Log statistics
        self.logger.info(f'Shape of expression matrix: {adata.shape}')
        self.logger.info(f'Shape of backbone network: {edge_index.shape}')
        if len(edge_weights) > 0:
            self.logger.info(f'Edge weights range: [{edge_weights.min():.4f}, {edge_weights.max():.4f}]')

        # 5. Save results to adata.uns
        adata.uns[self.out] = edge_index
        adata.uns['thres'] = self.thres
        # Store edge weights if they exist
        if len(edge_weights) > 0:
            adata.uns[f'{self.out}_weight'] = edge_weights
        
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