# anndata_similarity.py
# TODO translate notes
import re
import warnings
from typing import Callable, Dict, List, Optional

import anndata
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import yaml
from omegaconf import OmegaConf
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Suppress scipy warnings for constant input in Pearson correlation
warnings.filterwarnings("ignore", message="An input array is constant")


class AnnDataSimilarity:

    def __init__(self, adata1: anndata.AnnData, adata2: anndata.AnnData, cell_col: str,
                 ground_truth_conf_path: Optional[str] = None, adata1_name: Optional[str] = None,
                 adata2_name: Optional[str] = None,
                 methods=['cta_actinn', 'cta_celltypist', 'cta_scdeepsort', 'cta_singlecellnet'], tissue="blood"):
        """Initialize the AnnDataSimilarity object and perform data preprocessing."""
        self.adata1 = adata1.copy()
        self.adata2 = adata2.copy()
        self.origin_adata1 = adata1.copy()
        self.origin_adata2 = adata2.copy()
        self.cell_col = cell_col
        self.preprocess()
        self.results = {}
        self.results_score = {}
        self.ground_truth_conf_path = ground_truth_conf_path
        self.adata1_name = adata1_name
        self.adata2_name = adata2_name
        self.methods = methods
        self.tissue = tissue

    def filter_gene(self):
        sc.pp.highly_variable_genes(self.adata1, n_top_genes=2000, flavor='seurat_v3')
        sc.pp.highly_variable_genes(self.adata2, n_top_genes=2000, flavor='seurat_v3')

        common_hvg = self.adata1.var_names[self.adata1.var['highly_variable']].intersection(
            self.adata2.var_names[self.adata2.var['highly_variable']])

        self.adata1 = self.adata1[:, common_hvg].copy()
        self.adata2 = self.adata2[:, common_hvg].copy()
        self.common_genes = common_hvg

    def preprocess(self):
        self.filter_gene()
        """Preprocess the data, including log normalization and normalization to probability distribution."""
        self.adata1.obs[self.cell_col] = self.adata1.obs[self.cell_col].astype(str)
        self.adata2.obs[self.cell_col] = self.adata2.obs[self.cell_col].astype(str)
        self.avg_expr1 = self._compute_average_expression(self.adata1)
        self.avg_expr2 = self._compute_average_expression(self.adata2)
        self.prob_expr1 = self._normalize_to_probability(self.avg_expr1)
        self.prob_expr2 = self._normalize_to_probability(self.avg_expr2)

    def _compute_average_expression(self, adata: anndata.AnnData) -> pd.DataFrame:
        """Calculate the average gene expression for each cell type"""
        return adata.to_df().groupby(adata.obs[self.cell_col]).mean()

    def _normalize_to_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the gene expression matrix to a probability distribution (expression sums to 1 for each cell type)"""
        return df.div(df.sum(axis=1), axis=0).fillna(0)

    def cosine_sim(self) -> pd.DataFrame:
        """Computes the cosine similarity between two datasets. Returns a data frame with the cell types in rows and columns of adata1 and adata2 respectively."""
        sim_matrix = cosine_similarity(self.avg_expr1, self.avg_expr2)
        return pd.DataFrame(sim_matrix, index=self.avg_expr1.index, columns=self.avg_expr2.index)

    def pearson_corr(self) -> pd.DataFrame:
        """Computes the Pearson correlation coefficient between two datasets. Returns a data frame with the cell types in rows and columns of adata1 and adata2 respectively."""
        celltypes1 = self.avg_expr1.index
        celltypes2 = self.avg_expr2.index
        corr_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                corr, _ = pearsonr(self.avg_expr1.loc[ct1], self.avg_expr2.loc[ct2])
                corr_matrix.at[ct1, ct2] = corr

        return corr_matrix.astype(float)

    def jaccard_sim(self, threshold: float = 0.5) -> pd.DataFrame:
        """Computes the Jaccard similarity between two datasets. Uses a binary representation of gene expression based on a specified threshold. Returns a data frame with rows and columns of cell types in adata1 and adata2 respectively."""
        # Binarized expression matrix
        binary_expr1 = (self.avg_expr1 > threshold).astype(int)
        binary_expr2 = (self.avg_expr2 > threshold).astype(int)

        celltypes1 = binary_expr1.index
        celltypes2 = binary_expr2.index
        sim_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                sim = 1 - jaccard(binary_expr1.loc[ct1], binary_expr2.loc[ct2])
                sim_matrix.at[ct1, ct2] = sim

        return sim_matrix.astype(float)

    def js_distance(self) -> pd.DataFrame:
        """Computes the Jensen-Shannon divergence between two datasets. The expression data must first be normalized to a probability distribution. Returns a data frame with rows and columns containing the cell types of adata1 and adata2, respectively."""
        # def jsd(p, q):
        #     """
        #     计算两个概率分布 p 和 q 的 Jensen-Shannon 散度。
        #     """
        #     p = p + 1e-12
        #     q = q + 1e-12
        #     m = 0.5 * (p + q)
        #     return 0.5 * (entropy(p, m) + entropy(q, m))

        # from scipy.stats import entropy

        celltypes1 = self.prob_expr1.index
        celltypes2 = self.prob_expr2.index
        js_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                jsd_value = 1 - self._jensen_shannon_divergence(self.prob_expr1.loc[ct1].values,
                                                                self.prob_expr2.loc[ct2].values)
                js_matrix.at[ct1, ct2] = jsd_value

        return js_matrix.astype(float)

    def _jensen_shannon_divergence(self, p, q) -> float:
        """Compute the Jensen-Shannon divergence of two probability distributions p and q."""
        from scipy.spatial.distance import jensenshannon
        return jensenshannon(p, q)

    def common_genes_num(self):
        return len(self.common_genes)

    def otdd():
        """Compute the OTDD between two data sets."""
        raise NotImplementedError("OTDD!")

    def data_company():
        raise NotImplementedError("data company")

    def wasserstein_dist(self) -> pd.DataFrame:
        """Compute the Wasserstein distance between two datasets. Return a data frame with the cell types in rows and columns of adata1 and adata2 respectively."""
        celltypes1 = self.avg_expr1.index
        celltypes2 = self.avg_expr2.index
        wasserstein_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                wd = wasserstein_distance(self.avg_expr1.loc[ct1], self.avg_expr2.loc[ct2])
                wasserstein_matrix.at[ct1, ct2] = wd

        return wasserstein_matrix.astype(float)

    def get_dataset_meta_sim(self):
        # dis_cols=['assay', 'cell_type', 'development_stage','disease','is_primary_data','self_reported_ethnicity','sex', 'suspension_type', 'tissue','tissue_type', 'tissue_general']
        con_cols = [
            "nnz_mean", "nnz_var", "nnz_counts_mean", "nnz_counts_var", "n_measured_vars", "n_counts_mean",
            "n_counts_var", "var_n_counts_mean", "var_n_counts_var"
        ]
        dis_cols = ['assay', 'tissue']

        def get_discrete_sim(col_list1, col_list2):
            set1 = set(col_list1)
            set2 = set(col_list2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union

        def get_con_sim(con_data_1, con_data_2):
            return abs(con_data_1 - con_data_2) / max(con_data_1, con_data_2)

        def get_dataset_info(data: ad.AnnData):
            con_sim = {}
            con_sim["nnz_mean"] = np.mean(data.obs["nnz"])
            con_sim["nnz_var"] = np.var(data.obs["nnz"])
            nnz_values = data.X[data.X.nonzero()]
            con_sim["nnz_counts_mean"] = np.mean(nnz_values)
            con_sim["nnz_counts_var"] = np.var(nnz_values)
            con_sim["n_measured_vars"] = np.mean(data.obs["n_measured_vars"])
            con_sim["cell_num"] = len(data.obs)
            con_sim["gene_num"] = len(data.var)
            con_sim["n_counts_mean"] = np.mean(data.obs["n_counts"])
            con_sim["n_counts_var"] = np.var(data.obs["n_counts"])
            if "n_counts" not in data.var.columns:
                if scipy.sparse.issparse(data.X):
                    gene_counts = np.array(data.X.sum(axis=0)).flatten()
                else:
                    gene_counts = data.X.sum(axis=0)
            data.var["n_counts"]=gene_counts
            data.var["n_counts"]=data.var["n_counts"].astype(float)
            con_sim["var_n_counts_mean"] = np.mean(data.var["n_counts"])
            con_sim["var_n_counts_var"] = np.var(data.var["n_counts"])
            data.uns["con_sim"] = con_sim
            return data

        data_1 = self.adata1.copy()
        data_2 = self.adata2.copy()
        data_1 = get_dataset_info(data_1)
        data_2 = get_dataset_info(data_2)
        ans = {}
        obs_1 = data_1.obs
        obs_2 = data_2.obs
        con_sim_1 = data_1.uns["con_sim"]
        con_sim_2 = data_2.uns["con_sim"]
        for dis_col in dis_cols:
            ans[f"{dis_col}_sim"] = get_discrete_sim(obs_1[dis_col].values, obs_2[dis_col].values)
        for con_col in con_cols:
            ans[f"{con_col}_sim"] = get_con_sim(con_sim_1[con_col], con_sim_2[con_col])
        return np.mean(list(ans.values()))

    def get_ground_truth(self):
        assert self.ground_truth_conf_path is not None
        assert self.adata1_name is not None
        assert self.adata2_name is not None
        ground_truth_conf = pd.read_excel(self.ground_truth_conf_path, sheet_name=self.tissue, index_col=0)

        def get_targets(dataset_truth: str):
            dataset_truth = OmegaConf.create(fix_yaml_string(dataset_truth))
            targets = []
            for item in dataset_truth:
                targets.append(item["target"])
            return targets

        sim_targets = []
        for method in self.methods:
            query_dataset_truth = ground_truth_conf.loc[self.adata1_name, f"{method}_method"]
            atlas_dataset_truth = ground_truth_conf.loc[self.adata2_name, f"{method}_method"]
            query_targets = get_targets(query_dataset_truth)
            atlas_targets = get_targets(atlas_dataset_truth)
            assert len(query_targets) == len(atlas_targets)
            sim_targets.append((sum(a == b for a, b in zip(query_targets, atlas_targets)), len(query_targets)))
        sim_targets.append((sum(x for x, y in sim_targets), sum(y for x, y in sim_targets)))
        return sim_targets

    def compute_similarity(
        self, methods: List[str] = [
            'cosine', 'pearson', 'jaccard', 'js_distance', 'otdd', 'common_genes_num', "ground_truth", "metadata_sim"
        ]
    ) -> Dict[str, pd.DataFrame]:
        """Computes the specified similarity measure. Parameters:

        methods: List of similarity measures to be computed. Supports 'cosine', 'pearson', 'jaccard', 'js_distance', 'wasserstein','otdd'
        Returns:
        Dictionary containing the similarity matrices

        """
        results = {}
        for method in methods:
            if method == 'cosine':
                results['cosine'] = self.cosine_sim()
            elif method == 'pearson':
                results['pearson'] = self.pearson_corr()
            elif method == 'jaccard':
                results['jaccard'] = self.jaccard_sim()
            elif method == 'js_distance':
                results['js_distance'] = self.js_distance()
            elif method == 'wasserstein':
                results['wasserstein'] = self.wasserstein_dist()
            elif method == "common_genes_num":
                results["common_genes_num"] = self.common_genes_num()
            elif method == "otdd":
                results['otdd'] = self.otdd()
            elif method == "ground_truth":
                results["ground_truth"] = self.get_ground_truth()
            elif method == "metadata_sim":
                results["metadata_sim"] = self.get_dataset_meta_sim()
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
        return results

    def get_similarity_matrix(
        self, methods: List[str] = [
            'cosine', 'pearson', 'jaccard', 'js_distance', "common_genes_num", "ground_truth", "metadata_sim"
        ]
    ) -> Dict[str, pd.DataFrame]:
        """Same as compute_similarity, keeping method name consistency."""
        self.results = self.compute_similarity(methods)
        return self.results

    def get_max_similarity_A_to_B(self):
        if self.results is None:
            raise ValueError(f"need results!")
        else:
            self.results_score = {}
            for key in self.results:
                if key not in ["common_genes_num", "ground_truth", "metadata_sim"]:
                    self.results_score[key] = self._get_max_similarity(self.results[key])
                else:
                    self.results_score[key] = self.results[key]
        return self.results_score

    def _get_max_similarity(self, similarity_matrix: pd.DataFrame):
        """Maximum matching average similarity score."""
        matched_values = [
            similarity_matrix.loc[label,
                                  label] if label in similarity_matrix.columns else similarity_matrix.loc[label].max()
            for label in similarity_matrix.index
        ]  # need to ask
        overall_similarity = np.mean(matched_values)
        return overall_similarity


def extract_type_target_params(item_text):
    lines = item_text.strip().split('\n')
    item_dict = {}
    params_dict = {}
    current_param_key = None
    in_params = False
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('- type:'):
            item_dict['type'] = stripped_line.split(':', 1)[1].strip()
        elif stripped_line.startswith('target:'):
            item_dict['target'] = stripped_line.split(':', 1)[1].strip()
        elif stripped_line.startswith('params:'):
            params_content = stripped_line.split(':', 1)[1].strip()
            if params_content == '{}':
                params_dict = {}
                in_params = False
            else:
                params_dict = {}
                in_params = True
        elif in_params:
            if re.match(r'^\w+:$', stripped_line):
                current_param_key = stripped_line[:-1].strip()
                params_dict[current_param_key] = {}
            elif re.match(r'^- ', stripped_line):
                list_item = stripped_line[2:].strip()
                if current_param_key:
                    if not isinstance(params_dict[current_param_key], list):
                        params_dict[current_param_key] = []
                    params_dict[current_param_key].append(list_item)
            elif ':' in stripped_line:
                key, value = map(str.strip, stripped_line.split(':', 1))
                if current_param_key and isinstance(params_dict.get(current_param_key, None), dict):
                    params_dict[current_param_key][key] = yaml.safe_load(value)
                else:
                    params_dict[key] = yaml.safe_load(value)
    item_dict['params'] = params_dict
    return item_dict


def fix_yaml_string(original_str):
    #It will be deleted
    yaml_str = original_str.replace('\\n', '\n').strip()
    items = re.split(r'(?=-\s*type:)', yaml_str)
    config_list = []
    for item in items:
        if not item.strip():
            continue
        if not item.strip().startswith('- type:'):
            print(item)
            print("警告: 某个项未以 '- type:' 开头，跳过此项.")
            continue
        item_dict = extract_type_target_params(item)
        config_list.append(item_dict)
    fixed_yaml = yaml.dump(config_list, sort_keys=False)
    return fixed_yaml
