# anndata_similarity.py
# TODO translate notes
import warnings
from typing import Callable, Dict, List

import anndata
import numpy as np
import pandas as pd
from scipy.spatial.distance import jaccard
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

# Suppress scipy warnings for constant input in Pearson correlation
warnings.filterwarnings("ignore", message="An input array is constant")


class AnnDataSimilarity:

    def __init__(self, adata1: anndata.AnnData, adata2: anndata.AnnData):
        """初始化 AnnDataSimilarity 对象，进行数据预处理。"""
        self.adata1 = adata1.copy()
        self.adata2 = adata2.copy()
        self.preprocess()
        self.results = {}
        self.results_score = {}

    def preprocess(self):
        """预处理数据，包括对数归一化和归一化为概率分布。"""
        # 对原始数据进行对数归一化
        self.adata1.obs['celltype'] = self.adata1.obs['celltype'].astype(str)
        self.adata2.obs['celltype'] = self.adata2.obs['celltype'].astype(str)

        # 计算每个细胞类型的平均表达
        self.avg_expr1 = self._compute_average_expression(self.adata1)
        self.avg_expr2 = self._compute_average_expression(self.adata2)

        # 归一化为概率分布以计算 JS 散度等
        self.prob_expr1 = self._normalize_to_probability(self.avg_expr1)
        self.prob_expr2 = self._normalize_to_probability(self.avg_expr2)

    def _compute_average_expression(self, adata: anndata.AnnData) -> pd.DataFrame:
        """计算每种细胞类型的平均基因表达。"""
        return adata.to_df().groupby(adata.obs['celltype']).mean()

    def _normalize_to_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """将基因表达矩阵归一化为概率分布（每个细胞类型的表达总和为1）。"""
        return df.div(df.sum(axis=1), axis=0).fillna(0)

    def cosine_sim(self) -> pd.DataFrame:
        """计算两个数据集间的余弦相似度。 返回数据框，行和列分别为 adata1 和 adata2 的细胞类型。"""
        sim_matrix = cosine_similarity(self.avg_expr1, self.avg_expr2)
        return pd.DataFrame(sim_matrix, index=self.avg_expr1.index, columns=self.avg_expr2.index)

    def pearson_corr(self) -> pd.DataFrame:
        """计算两个数据集间的皮尔逊相关系数。 返回数据框，行和列分别为 adata1 和 adata2 的细胞类型。"""
        celltypes1 = self.avg_expr1.index
        celltypes2 = self.avg_expr2.index
        corr_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                corr, _ = pearsonr(self.avg_expr1.loc[ct1], self.avg_expr2.loc[ct2])
                corr_matrix.at[ct1, ct2] = corr

        return corr_matrix.astype(float)

    def jaccard_sim(self, threshold: float = 0.5) -> pd.DataFrame:
        """计算两个数据集间的 Jaccard 相似度。 使用基因表达的二值化表示，基于指定阈值。 返回数据框，行和列分别为 adata1 和 adata2
        的细胞类型。"""
        # 二值化表达矩阵
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
        """计算两个数据集间的 Jensen-Shannon 散度。 需要先将表达数据归一化为概率分布。 返回数据框，行和列分别为 adata1 和 adata2
        的细胞类型。"""
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
        """计算两个概率分布 p 和 q 的 Jensen-Shannon 散度。"""
        from scipy.spatial.distance import jensenshannon
        return jensenshannon(p, q)

    def otdd():
        """计算两个数据集间的 OTDD。"""
        raise NotImplementedError("OTDD!")

    def wasserstein_dist(self) -> pd.DataFrame:
        """计算两个数据集间的 Wasserstein 距离。 返回数据框，行和列分别为 adata1 和 adata2 的细胞类型。"""
        celltypes1 = self.avg_expr1.index
        celltypes2 = self.avg_expr2.index
        wasserstein_matrix = pd.DataFrame(index=celltypes1, columns=celltypes2)

        for ct1 in celltypes1:
            for ct2 in celltypes2:
                wd = wasserstein_distance(self.avg_expr1.loc[ct1], self.avg_expr2.loc[ct2])
                wasserstein_matrix.at[ct1, ct2] = wd

        return wasserstein_matrix.astype(float)

    def compute_similarity(
            self, methods: List[str] = ['cosine', 'pearson', 'jaccard', 'js_distance',
                                        'otdd']) -> Dict[str, pd.DataFrame]:
        """计算指定的相似性度量。 参数:

        methods: 要计算的相似性度量方法列表。支持 'cosine', 'pearson', 'jaccard', 'js_distance', 'wasserstein','otdd'
        返回:
            包含各个相似性矩阵的字典

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
            elif method == "otdd":
                results['otdd'] = self.otdd()
            else:
                raise ValueError(f"Unsupported similarity method: {method}")
        return results

    def get_similarity_matrix(
            self, methods: List[str] = ['cosine', 'pearson', 'jaccard', 'js_distance']) -> Dict[str, pd.DataFrame]:
        """同 compute_similarity，保留方法名一致性。"""
        self.results = self.compute_similarity(methods)
        return self.results

    def get_max_similarity_A_to_B(self):
        if self.results is None:
            raise ValueError(f"need results!")
        else:
            self.results_score = {}
            for key in self.results:
                self.results_score[key] = self._get_max_similarity(self.results[key])
        return self.results_score

    def _get_max_similarity(self, similarity_matrix: pd.DataFrame):
        """最大匹配平均相似性分数."""
        max_similarity = similarity_matrix.max(axis=1)
        overall_similarity = max_similarity.mean()
        return overall_similarity
