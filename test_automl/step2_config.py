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


def getFunConfig(selected_keys=["normalize", "gene_filter", "gene_dim_reduction"]):
    global pipline2fun_dict
    pipline2fun_dict = {key: pipline2fun_dict[key] for key in selected_keys}
    count = 1
    for _, pipline_values in pipline2fun_dict.items():
        count *= len(pipline_values['values'])
    return pipline2fun_dict, count
