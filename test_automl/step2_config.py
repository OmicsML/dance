import functools
import itertools

import wandb
from fun2code import fun2code_dict

from dance.transforms.misc import SetConfig

#TODO register more functions and add more examples
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
    },
    "cell_filter": {
        "values": ["filter_cell_by_count"]
    },
    "mask_name": {
        "values": ["cell_wise_mask_data", "mask_data"]
    },
    "gene_hold_out_name": {
        "values": ["gene_hold_out"]
    }
}  #Functions registered in the preprocessing process


def generate_combinations_with_required_elements(elements, required_elements=[]):
    optional_elements = [x for x in elements if x not in required_elements]

    # Sort optional elements in the same order as in the `elements` list
    optional_elements.sort(key=lambda x: elements.index(x))

    # Generate all possible combinations of optional elements
    optional_combinations = []
    for i in range(len(optional_elements) + 1):
        optional_combinations += list(itertools.combinations(optional_elements, i))

    # Combine required elements with optional combinations to get all possible combinations
    all_combinations = []
    for optional_combination in optional_combinations:
        all_combinations.append([x for x in elements if x in required_elements or x in optional_combination])
    return all_combinations


def getFunConfig(selected_keys=None):
    """Get the config that needs to be optimized and the number of rounds."""
    global pipline2fun_dict
    pipline2fun_dict_subset = {key: pipline2fun_dict[key] for key in selected_keys}
    print(pipline2fun_dict)
    count = 1
    for _, pipline_values in pipline2fun_dict_subset.items():
        count *= len(pipline_values['values'])
    return pipline2fun_dict_subset, count


def get_transforms(config=None, set_data_config=True, save_raw=False):
    """Obtain the Compose of the preprocessing function according to the preprocessing
    process."""
    if ("normalize" not in config.keys() or config.normalize
            != "log1p") and ("gene_filter" in config.keys() and config.gene_filter == "highly_variable_genes"):

        return None

    transforms = []
    for key in config.keys():
        if save_raw and key == "normalize":
            transforms.append(fun2code_dict["save_raw"])
        print(key, config[key])
        transforms.append(fun2code_dict[config[key]]) if key in pipline2fun_dict.keys() else None
    if save_raw and "normalize" not in config.keys():
        transforms.append(fun2code_dict["save_raw"])
    if set_data_config:
        data_config = {"label_channel": "cell_type"}
        if "gene_dim_reduction" in config.keys():
            data_config.update({"feature_channel": fun2code_dict[config.gene_dim_reduction].name})
        transforms.append(SetConfig(data_config))
    return transforms


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
            except Exception as e:  #Except, etc. are not necessarily needed in the code.
                print(f"{func.__name__}{args}\n==> {e}")
                raise e

        return wrapper

    return decorator


def setStep2(func=None, original_list=None, required_elements=[]):
    """Generate corresponding decorators for different preprocessing."""
    # all_combinations = [
    #     combo for i in range(len(original_list) + 1)
    #     for combo in generate_combinations_with_required_elements(original_list, i, required_elements=required_elements)
    # ]
    all_combinations = generate_combinations_with_required_elements(elements=original_list,
                                                                    required_elements=required_elements)
    generated_functions = []
    for s_key in all_combinations:
        s_list = list(s_key)
        print(s_list)
        decorator = sweepDecorator(selected_keys=s_list)
        generated_functions.append(decorator(func))
    return generated_functions


def log_in_wandb(config):
    """Decorator wrapped using wandb.It is used in train function."""

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
