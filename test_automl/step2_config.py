from fun2code import fun2code_dict

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


def get_preprocessing_pipeline(config=None):
    if ("normalize" not in config.keys() or config.normalize
            != "log1p") and ("gene_filter" in config.keys() and config.gene_filter == "highly_variable_genes"):

        return None
    transforms = []
    transforms.append(fun2code_dict[config.normalize]) if "normalize" in config.keys() else None
    transforms.append(fun2code_dict[config.gene_filter]) if "gene_filter" in config.keys() else None
    transforms.append(fun2code_dict[config.gene_dim_reduction]) if "gene_dim_reduction" in config.keys() else None
    data_config = {"label_channel": "cell_type"}
    if "gene_dim_reduction" in config.keys():
        data_config.update({"feature_channel": fun2code_dict[config.gene_dim_reduction].name})
    transforms.append(SetConfig(data_config))
    preprocessing_pipeline = Compose(*transforms, log_level="INFO")
    return preprocessing_pipeline
