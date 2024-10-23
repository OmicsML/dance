# test_anndata_similarity.py

import anndata
import numpy as np
import pandas as pd
from anndata_similarity import AnnDataSimilarity


def create_test_ann_data():
    # 定义基因和细胞类型
    genes = ['gene1', 'gene2']
    celltypes1 = ['A', 'B']
    celltypes2 = ['A', 'B']

    # 创建数据集1
    data1 = np.array([
        [10, 0],  # 细胞类型 A
        [0, 10]  # 细胞类型 B
    ])
    obs1 = pd.DataFrame({'celltype': celltypes1}, index=['cell1', 'cell2'])
    adata1 = anndata.AnnData(X=data1, obs=obs1, var=pd.DataFrame(index=genes))

    # 创建数据集2
    data2 = np.array([
        [10, 0],  # 细胞类型 A
        [10, 0]  # 细胞类型 B
    ])
    obs2 = pd.DataFrame({'celltype': celltypes2}, index=['cell3', 'cell4'])
    adata2 = anndata.AnnData(X=data2, obs=obs2, var=pd.DataFrame(index=genes))

    return adata1, adata2


def run_test_case():
    # 创建测试数据
    adata1, adata2 = create_test_ann_data()

    # 初始化相似性计算器
    similarity_calculator = AnnDataSimilarity(adata1, adata2)

    # 计算相似性
    similarity_matrices = similarity_calculator.compute_similarity(
        methods=['cosine', 'pearson', 'jaccard', 'js_distance'])

    # 预期结果
    expected_cosine = pd.DataFrame([[1.0, 1.0], [0.0, 0.0]], index=['A', 'B'], columns=['A', 'B'])

    expected_pearson = pd.DataFrame([[1.0, 1.0], [-1.0, -1.0]], index=['A', 'B'], columns=['A', 'B'])

    expected_jaccard = pd.DataFrame([[1.0, 1.0], [0.0, 0.0]], index=['A', 'B'], columns=['A', 'B'])

    expected_js = pd.DataFrame([[1.0, 1.0], [0.167445, 0.167445]], index=['A', 'B'], columns=['A', 'B'])

    # 打印结果
    print("Computed Cosine Similarity:")
    print(similarity_matrices['cosine'])
    print("\nExpected Cosine Similarity:")
    print(expected_cosine)

    print("\nComputed Pearson Correlation:")
    print(similarity_matrices['pearson'])
    print("\nExpected Pearson Correlation:")
    print(expected_pearson)

    print("\nComputed Jaccard Similarity:")
    print(similarity_matrices['jaccard'])
    print("\nExpected Jaccard Similarity:")
    print(expected_jaccard)

    print("\nComputed Jensen-Shannon distance:")
    print(similarity_matrices['js_distance'])
    print("\nExpected Jensen-Shannon distance:")
    print(expected_js)

    # 验证结果是否与预期一致
    assert similarity_matrices['cosine'].equals(expected_cosine), "Cosine similarity does not match expected values."
    assert similarity_matrices['pearson'].equals(
        expected_pearson), "Pearson correlation does not match expected values."
    assert similarity_matrices['jaccard'].equals(expected_jaccard), "Jaccard similarity does not match expected values."

    # 由于浮点数计算的精度问题，使用近似比较
    assert np.allclose(similarity_matrices['js_distance'], expected_js,
                       atol=1e-4), "JS distance does not match expected values."

    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    run_test_case()
