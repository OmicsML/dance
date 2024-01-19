pipline2fun_dict = {
    "normalize": {
        "values": ["normalize_total", "log1p", "scaleFeature", "scTransform"]
    },
    "gene_filter": {
        "values":
        ["filter_gene_by_count", "filter_gene_by_percentile", "highly_variable_genes", "Filter_gene_by_regress_score"]
    },
    "gene_dim_reduction": {
        "values": ["cell_svd", "cell_weighted_pca", "cell_pca"]
    }
}
