import functools
from itertools import combinations

from fun2code import fun2code_dict

import wandb
from dance.transforms.misc import Compose, SetConfig

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
}  #Functions registered in the preprocessing process


def getFunConfig(selected_keys=None):
    """Get the config that needs to be optimized and the number of rounds."""
    global pipline2fun_dict
    pipline2fun_dict_subset = {key: pipline2fun_dict[key] for key in selected_keys}
    count = 1
    for _, pipline_values in pipline2fun_dict_subset.items():
        count *= len(pipline_values['values'])
    return pipline2fun_dict_subset, count


def get_preprocessing_pipeline(config=None):
    """Obtain the Compose of the preprocessing function according to the preprocessing
    process."""
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


def sweepDecorator(selected_keys=None, project="pytorch-cell_type_annotation_ACTINN"):
    """Decorator for preprocessing configuration functions."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pipline2fun_dict, count = getFunConfig(selected_keys)
            parameters_dict = pipline2fun_dict
            try:
                sweep_config, train = func(parameters_dict)
                sweep_id = wandb.sweep(sweep_config, project=project)
                wandb.agent(sweep_id, train, count=count)
            except Exception as e:
                print(f"{func.__name__}{args}\n==> {e}")
                raise e

        return wrapper

    return decorator


def setStep2(func=None, original_list=None):
    """Generate corresponding decorators for different preprocessing."""
    all_combinations = [combo for i in range(1,
                                             len(original_list) + 1) for combo in combinations(original_list, i)] + [[]]
    generated_functions = []
    for s_key in all_combinations:
        s_list = list(s_key)
        decorator = sweepDecorator(selected_keys=s_list)
        generated_functions.append(decorator(func))
    return generated_functions


def log_in_wandb(config):
    """Decorator wrapped using wandb."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with wandb.init(config=config):
                    config_s = wandb.config
                    result = func(config_s, *args, **kwargs)
                    wandb.log(result)
            except Exception as e:
                print(f"{func.__name__}{args}\n==> {e}")
                raise e

        return wrapper

    return decorator
