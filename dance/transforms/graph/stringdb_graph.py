import argparse
import hashlib
import logging
import os
from abc import ABC, abstractmethod

import anndata as ad
import mygene
import numpy as np
import pandas as pd
import requests_cache

from dance.data import Data
from dance.registry import register_preprocessor
from dance.settings import DANCEPKGDIR
from dance.transforms.base import BaseTransform
from dance.typing import List, LogLevel, Tuple


@register_preprocessor("graph", "cell")
class StringDBGraph(BaseTransform):
    """Generates a gene-gene interaction graph for an AnnData object by filtering a
    reference network (e.g., STRING) and mapping it to the gene expression features.

    Features:
    - Filters reference graph by score quantile.
    - Maps Gene Symbols to Entrez IDs using MyGeneInfo.
    - Adds self-loops to the graph.
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

    """

    _DISPLAY_ATTRS: Tuple[str] = ('net_file', 'thres', 'species')

    def __init__(self, net_file=os.path.join(DANCEPKGDIR, "metadata", "STRINGDB.graph.csv"), thres: float = 0.99,
                 species: str = "human", out: str = "edge_index", log_level: LogLevel = "INFO"):
        super().__init__(out=out, log_level=log_level)
        self.net_file = net_file
        self.thres = thres
        self.species = species
        self.out = out

        # Validate threshold
        assert 0 <= thres <= 1, "quantile should be a float value in [0,1]."

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

        if 'score' in edge_df.columns:
            new_df['score'] = fill_value

        edge_df = pd.concat([edge_df, new_df], ignore_index=True)
        return edge_df

    def _map_graph_to_genes(self, graph_df: pd.DataFrame, gene_list: List[str]) -> np.ndarray:
        """Maps graph edges (Symbols/Entrez) to the indices of the provided
        gene_list."""
        # 1. Prepare Graph
        graph_edge_df = graph_df.copy()
        graph_edge_df.columns = ['node1', 'node2']
        graph_edge_df = graph_edge_df.astype(str)

        # 2. Prepare MyGene Query
        symbol_to_idx_dict = {g.strip(): idx for idx, g in enumerate(gene_list)}

        requests_cache.install_cache('mygene_cache', expire_after=3600 * 24 * 30, allowable_methods=['GET', 'POST'])
        mg = mygene.MyGeneInfo()

        self.logger.info(f"Querying MyGene info for {len(symbol_to_idx_dict)} genes...")
        res = mg.querymany(symbol_to_idx_dict.keys(), scopes='symbol,alias', fields='entrezgene', species=self.species,
                           verbose=False)

        # 3. Build Mapping (Entrez -> Index in gene_list)
        entrez_to_index_dict = {}
        for item in res:
            if 'entrezgene' in item and 'query' in item:
                entrez_id = str(item['entrezgene'])
                original_symbol = item['query']

                if original_symbol in symbol_to_idx_dict:
                    idx = symbol_to_idx_dict[original_symbol]
                    entrez_to_index_dict[entrez_id] = idx

        self.logger.info(
            f"Mapping complete: {len(symbol_to_idx_dict)} symbols mapped to {len(entrez_to_index_dict)} Entrez IDs.")

        # 4. Apply Mapping
        graph_edge_df['node1'] = graph_edge_df['node1'].map(entrez_to_index_dict)
        graph_edge_df['node2'] = graph_edge_df['node2'].map(entrez_to_index_dict)

        # Remove edges where nodes weren't found in the gene list
        graph_edge_df = graph_edge_df.dropna().astype(int)

        # 5. Add Self Loops
        graph_edge_df = self._add_remaining_self_loop(graph_edge_df, num_nodes=len(gene_list), fill_value=1.0)

        return graph_edge_df.values

    def __call__(self, data: Data) -> Data:
        """
        Process the AnnData object: load graph, filter, map, and update adata.uns.
        """
        self.logger.info(f"Processing data with threshold {self.thres}")
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
        filtered_graph = graph_df.loc[graph_df['score'].ge(cutoff_value), ['node1', 'node2']]

        self.logger.info(f"Graph filtered. Retained edges with score >= {cutoff_value:.4f}")

        # 2. Extract Gene Names and Process Graph
        # Check for gene names in .var_names or .var.index
        gene_names = adata.var_names.tolist()

        edge_index = self._map_graph_to_genes(filtered_graph, gene_names)

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

        # 5. Save results to adata.uns
        adata.uns[self.out] = edge_index
        adata.uns['thres'] = self.thres

        # If strict compatibility with `gen_data` output structure is required:
        # The original function returned a NEW adata.
        # Here we modify in place. If you strictly need the 'cleaned' structure
        # (just X, cell_type, var), you would do it here, but typically
        # keeping the full object is better.

        self.logger.info('Finished processing data.')
        return data


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()

    # Data generation parameters
    parser.add_argument('-h5ad', '--h5ad', type=str, help='Path to h5ad file containing expression data and labels')
    parser.add_argument('-net', '--net', type=str, help='Path to network CSV file')
    parser.add_argument('-q', '--quantile', type=float, default='0.99')

    # Training parameters
    parser.add_argument('-out_dir', '--out_dir', type=str, default='../results')
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)

    return parser


from dance.settings import DANCEPKGDIR


def gen_data(adata, net_file=os.path.join(DANCEPKGDIR, "metadata", "STRINGDB.graph.csv"), thres=0.99, species="human"):
    """Generate AnnData object from h5ad file and network file.

    Parameters:
    h5ad_file (str): Path to h5ad file containing expression data and labels
    net_file (str): Path to network CSV file
    thres (float): Quantile threshold for network filtering (default: 0.99)

    Returns:
    anndata.AnnData: AnnData object containing processed data

    """
    assert 0 <= thres <= 1, "quantile should be a float value in [0,1]."

    adata_raw = adata
    # Extract expression matrix (genes x cells)
    data_df = pd.DataFrame(adata_raw.X.T, index=adata_raw.var.index, columns=adata_raw.obs.index)

    # Extract labels
    label_df = pd.DataFrame({'cell_type': adata_raw.obs['cell_type']}, index=adata_raw.obs.index)

    graph_df = pd.read_csv(
        net_file,
        header=None,
        index_col=None,
    )
    graph_df.columns = ['node1', 'node2', 'score']
    graph_df = graph_df.loc[graph_df.score.ge(graph_df.score.quantile(thres)).values, ['node1', 'node2']]

    str_labels = np.unique(label_df.values).tolist()
    label = [str_labels.index(x) for x in label_df.values]
    gene = data_df.index.values
    barcode = data_df.columns.values
    edge_index = coding_edge_with_ref_gene_idx(graph_df.values, gene, species)

    print('shape of expression matrix [#genes,#cells]:', data_df.shape)
    print('shape of cell labels:', len(label))
    print('number of cell types:', len(str_labels))
    print('shape of backbone network:', edge_index.shape)

    # Create AnnData object
    adata = ad.AnnData(
        X=data_df.values.T,  # cells x genes
        obs=pd.DataFrame({'cell_type': [str_labels[i] for i in label]}, index=barcode),
        var=pd.DataFrame(index=gene))

    # Add additional information to uns
    adata.uns['str_labels'] = str_labels
    adata.uns['edge_index'] = edge_index
    adata.uns['thres'] = thres

    print('Finished processing data.')
    return adata


def add_remaining_self_loop_for_edge_df(edge_df, edge_weight_column='score', fill_value=1., num_nodes=None):
    """
    edge_df : #num_edges x 2
    """
    assert 'node1' in edge_df.columns
    assert 'node2' in edge_df.columns
    edge_index = edge_df[['node1', 'node2']].T.values
    row, col = edge_index[0], edge_index[1]
    N = num_nodes if num_nodes is not None else np.max(edge_index) + 1

    mask = row == col
    added_index = list(set(np.arange(0, N, dtype=int)) - set(row[mask]))

    new_df = pd.DataFrame()
    new_df['node1'] = added_index
    new_df['node2'] = added_index

    if edge_weight_column in edge_df.columns:
        new_df[edge_weight_column] = fill_value

    edge_df = pd.concat([edge_df, new_df], ignore_index=True)
    return edge_df


def coding_edge_with_ref_gene_idx(converted_graph_gene, converted_expr_gene, species='human'):
    """
    converted_graph_gene: #2 Dim
    converted_expr_gene: #1 Dim
    convert graph_gene to index which is the order of converted_expr_gene and drop nan
    """
    graph_edge_df = pd.DataFrame(converted_graph_gene, columns=['node1', 'node2'])
    graph_edge_df = graph_edge_df.astype(str)
    symbol_to_idx_dict = {g.strip(): idx for idx, g in enumerate(converted_expr_gene)}
    requests_cache.install_cache(
        'mygene_cache',
        expire_after=3600 * 24 * 30,  # 30天过期
        allowable_methods=['GET', 'POST']  # 关键：必须允许 POST，因为 querymany 是批量 POST
    )
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        symbol_to_idx_dict.keys(),
        scopes='symbol,alias',
        fields='entrezgene',
        species=species,  # 如果是小鼠改为 'mouse'
        verbose=False)
    entrez_to_index_dict = {}
    for item in res:
        # 必须同时存在查询到的 ID 和原始查询的 Symbol
        if 'entrezgene' in item and 'query' in item:
            entrez_id = str(item['entrezgene'])  # 转为字符串以匹配 DataFrame
            original_symbol = item['query']

            # 获取该 Symbol 在表达矩阵中的索引
            if original_symbol in symbol_to_idx_dict:
                idx = symbol_to_idx_dict[original_symbol]
                entrez_to_index_dict[entrez_id] = idx

    print(f"映射完成：{len(symbol_to_idx_dict)} 个 Symbol 中有 {len(entrez_to_index_dict)} 个成功匹配到 Entrez ID。")

    graph_edge_df['node1'] = graph_edge_df['node1'].map(entrez_to_index_dict)
    graph_edge_df['node2'] = graph_edge_df['node2'].map(entrez_to_index_dict)
    graph_edge_df = graph_edge_df.dropna().astype(int)

    graph_edge_df = add_remaining_self_loop_for_edge_df(graph_edge_df, edge_weight_column='score', fill_value=1,
                                                        num_nodes=len(converted_expr_gene))
    return graph_edge_df.values
