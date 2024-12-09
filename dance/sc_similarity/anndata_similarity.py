# anndata_similarity.py
# TODO translate notes
import re
import warnings
from typing import Callable, Dict, List, Optional

import anndata
import anndata as ad
import numpy as np
import ot
import pandas as pd
import scanpy as sc
import yaml
from omegaconf import OmegaConf
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, directed_hausdorff, jaccard, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

# Suppress scipy warnings for constant input in Pearson correlation
warnings.filterwarnings("ignore", message="An input array is constant")


class AnnDataSimilarity:

    def __init__(self, adata1: anndata.AnnData, adata2: anndata.AnnData, sample_size: Optional[int] = None,
                 init_random_state: Optional[int] = None, n_runs: int = 10,
                 ground_truth_conf_path: Optional[str] = None, adata1_name: Optional[str] = None,
                 adata2_name: Optional[str] = None,
                 methods=['cta_actinn', 'cta_celltypist', 'cta_scdeepsort', 'cta_singlecellnet'], tissue="blood"):
        """Initialize the AnnDataSimilarity object and perform data preprocessing."""
        self.origin_adata1 = adata1.copy()
        self.origin_adata2 = adata2.copy()
        self.sample_size = sample_size
        self.init_random_state = init_random_state
        self.preprocess()
        self.results = {}
        self.ground_truth_conf_path = ground_truth_conf_path
        self.adata1_name = adata1_name
        self.adata2_name = adata2_name
        self.methods = methods
        self.tissue = tissue
        self.n_runs = n_runs

    def filter_gene(self, n_top_genes=3000):
        sc.pp.highly_variable_genes(self.origin_adata1, n_top_genes=n_top_genes, flavor='seurat_v3')
        sc.pp.highly_variable_genes(self.origin_adata2, n_top_genes=n_top_genes, flavor='seurat_v3')

        common_hvg = self.origin_adata1.var_names[self.origin_adata1.var['highly_variable']].intersection(
            self.origin_adata2.var_names[self.origin_adata2.var['highly_variable']])

        self.origin_adata1 = self.origin_adata1[:, common_hvg].copy()
        self.origin_adata2 = self.origin_adata2[:, common_hvg].copy()
        self.common_genes = common_hvg

    def preprocess(self):
        """Preprocess the data, including log normalization and normalization to probability distribution."""
        self.filter_gene()

    def sample_cells(self, random_state):
        """
        Randomly sample cells from each dataset if sample_size is specified.
        """
        np.random.seed(random_state)
        if self.sample_size is None:
            self.sample_size = min(self.adata1.n_obs, self.adata2.n_obs)  #need to think
        if self.adata1.n_obs > self.sample_size:
            indices1 = np.random.choice(self.adata1.n_obs, size=self.sample_size, replace=False)
            self.sampled_adata1 = self.adata1[indices1, :].copy()
        else:
            self.sampled_adata1 = self.adata1.copy()
        if self.adata2.n_obs > self.sample_size:
            indices2 = np.random.choice(self.adata2.n_obs, size=self.sample_size, replace=False)
            self.sampled_adata2 = self.adata2[indices2, :].copy()
        else:
            self.sampled_adata2 = self.adata2.copy()

    def normalize_data(self):  # I am not sure
        """
        Normalize the data by total counts per cell and log-transform.
        """
        sc.pp.normalize_total(self.adata1, target_sum=1e4)
        sc.pp.log1p(self.adata1)
        sc.pp.normalize_total(self.adata2, target_sum=1e4)
        sc.pp.log1p(self.adata2)

    def set_prob_data(self, sampled=False):
        # Normalize the data to probability distributions
        if sampled:
            prob_adata1 = self.sampled_adata1.X / self.sampled_adata1.X.sum(axis=1)
            prob_adata2 = self.sampled_adata2.X / self.sampled_adata2.X.sum(axis=1)
        else:
            prob_adata1 = self.adata1.X / self.adata1.X.sum(axis=1)
            prob_adata2 = self.adata2.X / self.adata2.X.sum(axis=1)
        # Handle any NaN values resulting from division by zero
        self.X = np.nan_to_num(prob_adata1).toarray()
        self.Y = np.nan_to_num(prob_adata2).toarray()

    def cosine_sim_sampled(self) -> pd.DataFrame:
        """
        Computes the average cosine similarity between all pairs of cells from the two datasets.
        """
        # Compute cosine similarity matrix
        sim_matrix = cosine_similarity(self.sampled_adata1.X, self.sampled_adata2.X)
        # Return the average similarity
        return sim_matrix.mean()

    def pearson_corr_sampled(self) -> pd.DataFrame:
        """
        Computes the average Pearson correlation coefficient between all pairs of cells from the two datasets.
        """
        # Compute Pearson correlation matrix
        corr_matrix = np.corrcoef(self.sampled_adata1.X.toarray(),
                                  self.sampled_adata2.X.toarray())[:self.sampled_adata1.n_obs,
                                                                   self.sampled_adata1.n_obs:]
        # Return the average correlation
        return np.nanmean(corr_matrix)

    def jaccard_sim_sampled(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Computes the average Jaccard similarity between all pairs of binarized cells from the two datasets.
        """
        # Binarize the data
        binary_adata1 = (self.sampled_adata1.X > threshold).astype(int)
        binary_adata2 = (self.sampled_adata2.X > threshold).astype(int)
        # Compute Jaccard distance matrix
        distance_matrix = cdist(binary_adata1.A, binary_adata2.A, metric='jaccard')
        # Convert to similarity and compute the average
        similarity_matrix = 1 - distance_matrix
        return similarity_matrix.mean()

    def js_divergence_sampled(self) -> float:
        """
        Computes the average Jensen-Shannon divergence between all pairs of cells from the two datasets.
        """
        # Normalize the data to probability distributions
        prob_adata1 = self.sampled_adata1.X / self.sampled_adata1.X.sum(axis=1)
        prob_adata2 = self.sampled_adata2.X / self.sampled_adata2.X.sum(axis=1)
        # Handle any NaN values resulting from division by zero
        prob_adata1 = np.nan_to_num(prob_adata1).toarray()
        prob_adata2 = np.nan_to_num(prob_adata2).toarray()

        # Define a function to compute JS divergence for a pair of probability vectors
        def jsd(p, q):
            return jensenshannon(p, q)

        # Compute JS divergence matrix
        jsd_vectorized = np.vectorize(jsd, signature='(n),(n)->()')
        divergence_matrix = np.zeros((prob_adata1.shape[0], prob_adata2.shape[0]))
        for i in range(prob_adata1.shape[0]):
            divergence_matrix[i, :] = jsd_vectorized(
                np.repeat(prob_adata1[i, :], prob_adata2.shape[0], axis=0).reshape(-1, prob_adata1.shape[1]),
                prob_adata2)

        # Convert divergence to similarity and compute the average
        similarity_matrix = 1 - divergence_matrix
        return np.nanmean(similarity_matrix)

    def compute_mmd(self) -> float:
        X = self.X
        Y = self.Y
        kernel = "rbf"
        gamma = 1.0
        if kernel == 'rbf':
            K_X = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
            K_Y = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
            K_XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
        elif kernel == 'linear':
            K_X = np.dot(X, X.T)
            K_Y = np.dot(Y, Y.T)
            K_XY = np.dot(X, Y.T)
        else:
            raise ValueError("Unsupported kernel type")

        m = X.shape[0]
        n = Y.shape[0]

        sum_X = (np.sum(K_X) - np.sum(np.diag(K_X))) / (m * (m - 1))
        sum_Y = (np.sum(K_Y) - np.sum(np.diag(K_Y))) / (n * (n - 1))
        sum_XY = np.sum(K_XY) / (m * n)

        mmd_squared = sum_X + sum_Y - 2 * sum_XY
        mmd = np.sqrt(max(mmd_squared, 0))
        return 1 / (1 + mmd)

    def common_genes_num(self):
        return len(self.common_genes)

    def otdd(self):
        """Compute the OTDD between two data sets."""
        raise NotImplementedError("OTDD!")

    def data_company(self):
        raise NotImplementedError("data company")

    def wasserstein_dist(self) -> float:
        """
        Computes the average Wasserstein distance between all pairs of cells from the two datasets.
        """
        X = self.X
        Y = self.Y
        a = np.ones((X.shape[0], )) / X.shape[0]
        b = np.ones((Y.shape[0], )) / Y.shape[0]
        M = ot.dist(X, Y, metric='euclidean')
        wasserstein_dist = ot.emd2(a, b, M)
        return 1 / 1 + wasserstein_dist

    def get_Hausdorff(self):
        X = self.X
        Y = self.Y
        forward = directed_hausdorff(X, Y)[0]
        backward = directed_hausdorff(X, Y)[0]
        hausdorff_distance = max(forward, backward)
        normalized_hausdorff = hausdorff_distance / np.sqrt(X.shape[1])
        similarity = 1 - normalized_hausdorff
        return similarity

    def chamfer_distance(self):
        X = self.X
        Y = self.Y
        tree_A = cKDTree(X)
        tree_B = cKDTree(Y)

        distances_A_to_B, _ = tree_A.query(Y)
        distances_B_to_A, _ = tree_B.query(X)

        chamfer_A_to_B = np.mean(distances_A_to_B)
        chamfer_B_to_A = np.mean(distances_B_to_A)
        distance = chamfer_A_to_B + chamfer_B_to_A
        normalized_chamfer = distance / np.sqrt(X.shape[1])
        similarity = 1 - normalized_chamfer
        return similarity

    def energy_distance_metric(self):
        X = self.X
        Y = self.Y
        XX = cdist(X, X, 'euclidean')
        YY = cdist(Y, Y, 'euclidean')
        XY = cdist(X, Y, 'euclidean')
        distance = 2 * np.mean(XY) - np.mean(XX) - np.mean(YY)
        return 1 / (1 + distance)

    def get_sinkhorn2(self):
        X = self.X
        Y = self.Y
        a = np.ones(X.shape[0]) / X.shape[0]
        b = np.ones(Y.shape[0]) / Y.shape[0]
        M = ot.dist(X, Y, metric='euclidean')
        reg = 0.1
        sinkhorn_dist = ot.sinkhorn2(a, b, M, reg)
        return 1 / (1 + sinkhorn_dist)

    def bures_distance(self):
        X = self.X
        Y = self.Y
        C1 = np.cov(X, rowvar=False)
        C2 = np.cov(Y, rowvar=False)
        sqrt_C1 = sqrtm(C1)
        product = sqrt_C1 @ C2 @ sqrt_C1
        sqrt_product = sqrtm(product)
        trace = np.trace(C1) + np.trace(C2) - 2 * np.trace(sqrt_product)
        return 1 / (1 + np.sqrt(max(trace, 0)))

    def spectral_distance(self):
        X = self.X
        Y = self.Y
        C1 = np.cov(X, rowvar=False)
        C2 = np.cov(Y, rowvar=False)
        eig_A = np.linalg.eigvalsh(C1)
        eig_B = np.linalg.eigvalsh(C2)
        return 1 / (1 + np.linalg.norm(eig_A - eig_B))

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
            con_sim["nnz_mean"] = np.mean(data.obs["nnz"])  #sample 10000之后这里是应该更新的
            con_sim["nnz_var"] = np.var(data.obs["nnz"])
            nnz_values = data.X[data.X.nonzero()]
            con_sim["nnz_counts_mean"] = np.mean(nnz_values)
            con_sim["nnz_counts_var"] = np.var(nnz_values)
            con_sim["n_measured_vars"] = np.mean(data.obs["n_measured_vars"])
            con_sim["cell_num"] = len(data.obs)
            con_sim["gene_num"] = len(data.var)
            con_sim["n_counts_mean"] = np.mean(data.obs["n_counts"])
            con_sim["n_counts_var"] = np.var(data.obs["n_counts"])
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
        self, random_state: int, methods: List[str] = [
            'cosine', 'pearson', 'jaccard', 'js_distance', 'otdd', 'common_genes_num', "ground_truth", "metadata_sim"
        ]
    ) -> Dict[str, float]:
        """Computes the specified similarity measure. Parameters:

        methods: List of similarity measures to be computed. Supports 'cosine', 'pearson', 'jaccard', 'js_distance', 'wasserstein','otdd'
        Returns:
        Dictionary containing the similarity matrices

        """
        self.adata1 = self.origin_adata1.copy()
        self.adata2 = self.origin_adata2.copy()
        self.normalize_data()
        self.sample_cells(random_state)
        self.set_prob_data()

        results = {}
        for method in methods:
            print(method)
            if method == 'cosine':
                results['cosine'] = self.cosine_sim_sampled()
            elif method == 'pearson':
                results['pearson'] = self.pearson_corr_sampled()
            elif method == 'jaccard':
                results['jaccard'] = self.jaccard_sim_sampled()
            elif method == 'js_distance':
                results['js_distance'] = self.js_divergence_sampled()
            elif method == 'wasserstein':
                results['wasserstein'] = self.wasserstein_dist()
            elif method == "common_genes_num":
                results["common_genes_num"] = self.common_genes_num()
            elif method == "Hausdorff":
                results["Hausdorff"] = self.get_Hausdorff()
            elif method == "chamfer":
                results["chamfer"] = self.chamfer_distance()
            elif method == "energy":
                results["energy"] = self.energy_distance_metric()
            elif method == "sinkhorn2":
                results["sinkhorn2"] = self.get_sinkhorn2()
            elif method == "bures":
                results["bures"] = self.bures_distance()
            elif method == "spectral":
                results["spectral"] = self.spectral_distance()
            elif method == "otdd":
                results['otdd'] = self.otdd()
            elif method == "ground_truth":
                results["ground_truth"] = self.get_ground_truth()
            elif method == "metadata_sim":
                results["metadata_sim"] = self.get_dataset_meta_sim()
            elif method == "mmd":
                results["mmd"] = self.compute_mmd()
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
        return results

    def get_similarity_matrix_A2B(
        self, methods: List[str] = [
            "wasserstein", "Hausdorff", "chamfer", "energy", "sinkhorn2", "bures", "spectral", "common_genes_num",
            "ground_truth", "metadata_sim", "mmd"
        ]
    ) -> Dict[str, float]:
        """Same as compute_similarity, keeping method name consistency."""
        cumulative_results = {method: 0.0 for method in methods}

        for run in range(self.n_runs):
            # Update random state for each run
            if self.init_random_state is not None:
                current_random_state = self.init_random_state + run
            else:
                current_random_state = None
            run_results = self.compute_similarity(methods=methods, random_state=current_random_state)
            for method in methods:
                if method in ["ground_truth"]:
                    cumulative_results[method] = run_results[method]
                else:
                    cumulative_results[method] += run_results[method]
    # Average the results over the number of runs
        averaged_results = {
            method:
            cumulative_results[method] if method in ["ground_truth"] else cumulative_results[method] / self.n_runs
            for method in methods
        }
        return averaged_results

    # def get_max_similarity_A_to_B(self):
    #     if self.results is None:
    #         raise ValueError(f"need results!")
    #     else:
    #         self.results_score = {}
    #         for key in self.results:
    #             if key not in ["common_genes_num", "ground_truth", "metadata_sim"]:
    #                 self.results_score[key] = self._get_max_similarity(self.results[key])
    #             else:
    #                 self.results_score[key] = self.results[key]
    #     return self.results_score

    # def _get_max_similarity(self, similarity_matrix: pd.DataFrame):
    #     """Maximum matching average similarity score."""
    #     matched_values = [
    #         similarity_matrix.loc[label,
    #                               label] if label in similarity_matrix.columns else similarity_matrix.loc[label].max()
    #         for label in similarity_matrix.index
    #     ]  # need to ask
    #     overall_similarity = np.mean(matched_values)
    #     return overall_similarity


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
