import scanpy as sc

from dance.transforms.cell_feature import CellPCA, CellSVD, WeightedFeaturePCA
from dance.transforms.filter import FilterGenesPercentile, FilterGenesRegression
from dance.transforms.interface import AnnDataTransform
from dance.transforms.normalize import ScaleFeature, ScTransformR

#TODO register more functions
fun2code_dict = {
    "normalize_total": AnnDataTransform(sc.pp.normalize_total, target_sum=1e4),
    "log1p": AnnDataTransform(sc.pp.log1p, base=2),
    "scaleFeature": ScaleFeature(split_names="ALL", mode="standardize"),
    "scTransform": ScTransformR(mirror_index=1),
    "filter_gene_by_count": AnnDataTransform(sc.pp.filter_genes, min_cells=1),
    "filter_gene_by_percentile": FilterGenesPercentile(min_val=1, max_val=99, mode="sum"),
    "highly_variable_genes": AnnDataTransform(sc.pp.highly_variable_genes),
    "regress_out": AnnDataTransform(sc.pp.regress_out),
    "Filter_gene_by_regress_score": FilterGenesRegression("enclasc"),
    "cell_svd": CellSVD(),
    "cell_weighted_pca": WeightedFeaturePCA(split_name="train"),
    "cell_pca": CellPCA(),
    # "filter_cell_by_count":AnnDataTransform(sc.pp.filter_cells,min_genes=1)
}  #funcion 2 code
