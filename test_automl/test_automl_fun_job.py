import os
import random
import sys

import numpy as np
import scanpy as sc
import torch
import wandb
from optuna_wandb import fun_list

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.actinn import ACTINN
from dance.transforms.cell_feature import CellPCA, CellSVD, WeightedFeaturePCA
from dance.transforms.filter import FilterGenesPercentile, FilterGenesRegression
from dance.transforms.interface import AnnDataTransform
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import ScaleFeature, ScTransformR
from dance.utils import set_seed
from test_automl.wandb_step2 import fun2code_dict

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# import inspect
# path= os.path.dirname(inspect.getfile(lambda: None))
# path = os.path.join(path,"fun_list.json") # 拼接文件的路径
# with open(path,"r") as file:
#     loaded_data = json.load(file)
# fun_list = loaded_data["fun_list"]
def cell_pca(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return CellPCA(n_components=parameter_config.get(method_name + "n_components"))


def cell_weighted_pca(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return WeightedFeaturePCA(n_components=parameter_config.get(method_name + "n_components"))


def cell_svd(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return CellSVD(n_components=parameter_config.get(method_name + "n_components"))


def Filter_gene_by_regress_score(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return FilterGenesRegression(method=parameter_config.get(method_name + "method"),
                                 num_genes=parameter_config.get(method_name + "num_genes"))


def highly_variable_genes(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return AnnDataTransform(sc.pp.highly_variable_genes, min_mean=parameter_config.get(method_name + "min_mean"),
                            max_mean=parameter_config.get(method_name + "min_mean"),
                            min_disp=parameter_config.get(method_name + "min_disp"),
                            span=parameter_config.get(method_name + "span"),
                            n_bins=parameter_config.get(method_name + "n_bins"),
                            flavor=parameter_config.get(method_name + "flavor"))


def filter_gene_by_percentile(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return FilterGenesPercentile(min_val=parameter_config.get(method_name + "min_val"),
                                 max_val=parameter_config.get(method_name + "max_val"),
                                 mode=parameter_config.get(method_name + "mode"))


def filter_gene_by_count(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    method = parameter_config.get(method_name + "method")
    num = parameter_config.get(method_name + "num")
    return AnnDataTransform(sc.pp.filter_genes, **{method: num})


def log1p(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    base = parameter_config.get(method_name + "base")
    return AnnDataTransform(sc.pp.log1p, base=base)


def scTransform(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return ScTransformR(min_cells=parameter_config.get(method_name + "min_cells"))


def scaleFeature(parameter_config):  #eps未优化
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return ScaleFeature(mode=parameter_config.get(method_name + "mode", ["normalize", "standardize", "minmax", "l2"]))


def normalize_total(parameter_config):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    exclude_highly_expressed = parameter_config.get(method_name + "exclude_highly_expressed", [False, True])
    if exclude_highly_expressed:
        max_fraction = parameter_config.get(method_name + "max_fraction")
        return AnnDataTransform(sc.pp.normalize_total, target_sum=parameter_config.get(method_name + "target_sum"),
                                exclude_highly_expressed=exclude_highly_expressed, max_fraction=max_fraction)
    else:
        return AnnDataTransform(sc.pp.normalize_total, target_sum=parameter_config.get(method_name + "target_sum"),
                                exclude_highly_expressed=exclude_highly_expressed)


def run_training_run():
    settings = wandb.Settings(job_source="artifact")
    run = wandb.init(
        project="pytorch-cell_type_annotation_ACTINN_function_new",
        settings=settings,
        entity="xzy11632",
    )

    print("parameter_config")
    parameter_config = dict(run.config)
    print(parameter_config)
    print("fun_list" + str(fun_list))

    if len(parameter_config.keys()) != 0:

        parameters_dict = {
            'batch_size': 128,
            "hidden_dims": [2000],
            'lambd': 0.005,
            'num_epochs': 50,
            'seed': 0,
            'num_runs': 1,
            'learning_rate': 0.0001
        }
        parameter_config.update(parameters_dict)
        transforms = []
        for f_str in fun_list:
            fun_i = eval(f_str)
            transforms.append(fun_i(parameter_config))
        print(transforms)
        data_config = {"label_channel": "cell_type"}
        feature_name = {"cell_svd", "cell_weighted_pca", "cell_pca"} & set(fun_list)
        if feature_name:
            data_config.update({"feature_channel": fun2code_dict[feature_name].name})

        transforms.append(SetConfig(data_config))
        preprocessing_pipeline = Compose(*transforms, log_level="INFO")

        # preprocessing_pipeline = model.preprocessing_pipeline(normalize=args.normalize, filter_genes=not args.nofilter)
        # Load data and perform necessary preprocessing
        train_dataset = [753, 3285]
        test_dataset = [2695]
        tissue = "Brain"
        species = "mouse"
        dataloader = CellTypeAnnotationDataset(train_dataset=train_dataset, test_dataset=test_dataset, tissue=tissue,
                                               species=species)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=True)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data(return_type="torch")
        x_test, y_test = data.get_test_data(return_type="torch")
        x_train, y_train, x_test, y_test = x_train.float(), y_train.float(), x_test.float(), y_test.float()
        # Train and evaluate models for several rounds
        scores = []
        model = ACTINN(hidden_dims=parameter_config.get('hidden_dims'), lambd=parameter_config.get('lambd'),
                       device=device)
        for seed in range(parameter_config.get('seed'),
                          parameter_config.get('seed') + parameter_config.get('num_runs')):
            set_seed(seed)

            model.fit(x_train, y_train, seed=seed, lr=parameter_config.get('learning_rate'),
                      num_epochs=parameter_config.get('num_epochs'), batch_size=parameter_config.get('batch_size'),
                      print_cost=False)
            scores.append(score := model.score(x_test, y_test))
        #     print(f"{score=:.4f}")
        # print(f"ACTINN {species} {tissue} {test_dataset}:")
        # print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
        wandb.log({"scores": np.mean(scores)})
    run.log_code()
    run.finish()


run_training_run()
