import functools
import sys

import optuna
import scanpy as sc
import wandb
from fun2code import fun2code_dict
from optuna.integration.wandb import WeightsAndBiasesCallback
from step2_config import pipline2fun_dict

from dance.transforms.cell_feature import CellPCA, CellSVD, WeightedFeaturePCA
from dance.transforms.filter import FilterGenesPercentile, FilterGenesRegression
from dance.transforms.interface import AnnDataTransform
from dance.transforms.mask import CellwiseMaskData, MaskData
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import ScaleFeature, ScTransformR


def set_method_name(func):
    """Get method name to name the optimization option."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            method_name = func.__name__ + "_"
            result = func(method_name, *args, **kwargs)
            return result
        except Exception as e:
            print(f"{func.__name__}{args}\n==> {e}")
            raise e

    return wrapper


@set_method_name
def cell_pca(method_name: str, trial: optuna.Trial):
    return CellPCA(n_components=trial.suggest_int(method_name + "n_components", 200, 5000))


@set_method_name
def cell_weighted_pca(method_name: str, trial: optuna.Trial):
    return WeightedFeaturePCA(n_components=trial.suggest_int(method_name + "n_components", 200, 5000))


@set_method_name
def cell_svd(method_name: str, trial: optuna.Trial):
    return CellSVD(n_components=trial.suggest_int(method_name + "n_components", 200, 5000))


@set_method_name
def Filter_gene_by_regress_score(method_name: str, trial: optuna.Trial):
    return FilterGenesRegression(
        method=trial.suggest_categorical(method_name + "method", ["enclasc", "seurat3", "scmap"]),
        num_genes=trial.suggest_int(method_name + "num_genes", 5000, 6000))


@set_method_name
def highly_variable_genes(method_name: str, trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return AnnDataTransform(sc.pp.highly_variable_genes, min_mean=trial.suggest_float(
        method_name + "min_mean", 0.0025,
        0.03), max_mean=trial.suggest_float(method_name + "min_mean", 1.5,
                                            4.5), min_disp=trial.suggest_float(method_name + "min_disp", 0.25, 0.75),
                            span=trial.suggest_float(method_name + "span", 0.2,
                                                     1.0), n_bins=trial.suggest_int(method_name + "n_bins", 10, 30),
                            flavor=trial.suggest_categorical(method_name + "flavor", ['seurat', 'cell_ranger']))


@set_method_name
def filter_gene_by_percentile(method_name: str, trial: optuna.Trial):
    return FilterGenesPercentile(min_val=trial.suggest_int(method_name + "min_val", 1, 10),
                                 max_val=trial.suggest_int(method_name + "max_val", 90, 99),
                                 mode=trial.suggest_categorical(method_name + "mode", ["sum", "var", "cv", "rv"]))


@set_method_name
def filter_gene_by_count(method_name: str, trial: optuna.Trial):
    method = trial.suggest_categorical(method_name + "method", ['min_counts', 'min_cells', 'max_counts', 'max_cells'])
    if method == "min_counts":
        num = trial.suggest_int(method_name + "num", 2, 10)
    if method == "min_cells":
        num = trial.suggest_int(method_name + "num", 2, 10)
    if method == "max_counts":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    if method == "max_cells":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    return AnnDataTransform(sc.pp.filter_genes, **{method: num})


@set_method_name
def log1p(method_name: str, trial: optuna.Trial):
    return AnnDataTransform(sc.pp.log1p, base=trial.suggest_int(method_name + "base", 2, 10))


@set_method_name
def scTransform(method_name: str, trial: optuna.Trial):
    return ScTransformR(min_cells=trial.suggest_int(method_name + "min_cells", 1, 10))


@set_method_name
def scaleFeature(method_name: str, trial: optuna.Trial):  #eps未优化
    return ScaleFeature(mode=trial.suggest_categorical(method_name +
                                                       "mode", ["normalize", "standardize", "minmax", "l2"]))


@set_method_name
def normalize_total(method_name: str, trial: optuna.Trial):
    exclude_highly_expressed = trial.suggest_categorical(method_name + "exclude_highly_expressed", [False, True])
    if exclude_highly_expressed:
        max_fraction = trial.suggest_float(method_name + "max_fraction", 0.08, 0.1)
        return AnnDataTransform(sc.pp.normalize_total,
                                target_sum=trial.suggest_categorical(method_name + "target_sum", [1e4, 1e5, 1e6]),
                                exclude_highly_expressed=exclude_highly_expressed, max_fraction=max_fraction,
                                key_added="n_counts")
    else:
        return AnnDataTransform(sc.pp.normalize_total,
                                target_sum=trial.suggest_categorical(method_name + "target_sum", [1e4, 1e5, 1e6]),
                                exclude_highly_expressed=exclude_highly_expressed, key_added="n_counts")


@set_method_name
def filter_cell_by_count(method_name: str, trial: optuna.Trial):
    method = trial.suggest_categorical(method_name + "method", ['min_counts', 'min_genes', 'max_counts', 'max_genes'])
    if method == "min_counts":
        num = trial.suggest_int(method_name + "num", 1, 10)
    if method == "min_genes":
        num = trial.suggest_int(method_name + "num", 1, 10)
    if method == "max_counts":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    if method == "max_genes":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    return AnnDataTransform(sc.pp.filter_cells, **{method: num})


@set_method_name
def cell_wise_mask_data(method_name: str, trial: optuna.Trial):
    return CellwiseMaskData(distr=trial.suggest_categorical(method_name + "distr", ['exp', 'uniform']),
                            mask_rate=trial.suggest_float(method_name + "mask_rate", 0.01, 0.5),
                            min_gene_counts=trial.suggest_int(method_name + "min_gene_counts", 1, 10))


@set_method_name
def mask_data(method_name: str, trial: optuna.Trial):
    return MaskData(mask_rate=trial.suggest_float(method_name + "mask_rate", 0.01, 0.5))


# # 获取当前文件中的所有函数
# functions = [(name,obj) for name, obj in inspect.getmembers(
#     sys.modules[__name__]) if inspect.isfunction(obj)]

# print(functions)
# # 遍历并装饰每个函数
# for name, function in functions:
#     if name != "set_method_name":  # 排除装饰器函数本身
#         print(function)
#         setattr(__name__, name, set_method_name(function))


def get_transforms(trial, fun_list, set_data_config=True, save_raw=False):
    """Obtain the Compose of the preprocessing function according to the preprocessing
    function."""
    transforms = []
    for f_str in fun_list:
        if f_str in pipline2fun_dict['normalize']['values'] and save_raw:
            transforms.append(fun2code_dict["save_raw"])
        fun_i = eval(f_str)
        transforms.append(fun_i(trial))
    if "highly_variable_genes" in fun_list and "log1p" not in fun_list[:fun_list.index('"highly_variable_genes"')]:
        return None
    if set_data_config:
        data_config = {"label_channel": "cell_type"}
        feature_name = {"cell_svd", "cell_weighted_pca", "cell_pca"} & set(fun_list)
        if feature_name:
            data_config.update({"feature_channel": fun2code_dict[feature_name].name})
        transforms.append(SetConfig(data_config))
    return transforms


def log_in_wandb(wandbc=None):
    """Decorate optimization functions."""

    def decorator(func):

        def wrapper(*args, **kwargs):
            wandb_decorator = wandbc.track_in_wandb()
            decorator_function = wandb_decorator(func)
            result = decorator_function(*args, **kwargs)
            wandb.log(result)
            values = list(result.values())
            if len(values) == 1:
                return values[0]
            else:
                return tuple(values)

        return wrapper

    return decorator


def get_optimizer(project, objective, n_trials=2):
    """Get optimizer."""
    wandb_kwargs = {"project": project}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)
    decorator = log_in_wandb(wandbc)
    decorator_function = decorator(objective)
    study = optuna.create_study()

    def wrapper():
        study.optimize(decorator_function, n_trials=n_trials, callbacks=[wandbc])

    return wrapper
