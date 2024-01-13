import json
import os
import sys

import optuna
import scanpy as sc

fun_list = ["log1p", "filter_gene_by_count"]
# import inspect

# path= os.path.dirname(inspect.getfile(lambda: None))
# path = os.path.join(path,"fun_list.json") # 拼接文件的路径
# with open(path,"r") as file:
#     loaded_data = json.load(file)
# fun_list = loaded_data["fun_list"]

# print(path)
# from fun_list import fun_list
# from dance.transforms.cell_feature import CellPCA,CellSVD,WeightedFeaturePCA
# from dance.transforms.filter import FilterGenesPercentile, FilterGenesRegression
# from dance.transforms.interface import AnnDataTransform
# from dance.transforms.normalize import ScTransformR, ScaleFeature


def cell_pca(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return {method_name + "n_components": trial.suggest_int(method_name + "n_components", 200, 5000)}


    # return CellPCA(n_components=trial.suggest_int(method_name+"n_components",200,5000))
def cell_weighted_pca(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return {method_name + "n_components": trial.suggest_int(method_name + "n_components", 200, 5000)}


    # return WeightedFeaturePCA(n_components=trial.suggest_int(method_name+"n_components",200,5000))
def cell_svd(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return {method_name + "n_components": trial.suggest_int(method_name + "n_components", 200, 5000)}


    # return CellSVD(n_components=trial.suggest_int(method_name+"n_components",200,5000))
def Filter_gene_by_regress_score(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    # return FilterGenesRegression(method=trial.suggest_categorical(method_name+"method",["enclasc","seurat3","scmap"]),
    #                              num_genes=trial.suggest_int(method_name+"num_genes",5000,6000))
    return {
        method_name + "method": trial.suggest_categorical(method_name + "method", ["enclasc", "seurat3", "scmap"]),
        method_name + "num_genes": trial.suggest_int(method_name + "num_genes", 5000, 6000)
    }


def highly_variable_genes(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    # return AnnDataTransform(sc.pp.highly_variable_genes,min_mean=trial.suggest_float(method_name+"min_mean",0.0025,0.03),
    #                         max_mean=trial.suggest_float(method_name+"min_mean",1.5,4.5),
    #                         min_disp=trial.suggest_float(method_name+"min_disp",0.25,0.75),
    #                         span=trial.suggest_float(method_name+"span",0.2,1.0),
    #                         n_bins=trial.suggest_int(method_name+"n_bins",10,30),
    #                         flavor=trial.suggest_categorical(method_name+"flavor",['seurat', 'cell_ranger'])
    #                         )
    return {
        method_name + "min_mean": trial.suggest_float(method_name + "min_mean", 0.0025, 0.03),
        method_name + "max_mean": trial.suggest_float(method_name + "min_mean", 1.5, 4.5),
        method_name + "min_disp": trial.suggest_float(method_name + "min_disp", 0.25, 0.75),
        method_name + "span": trial.suggest_float(method_name + "span", 0.2, 1.0),
        method_name + "n_bins": trial.suggest_int(method_name + "n_bins", 10, 30),
        method_name + "flavor": trial.suggest_categorical(method_name + "flavor", ['seurat', 'cell_ranger'])
    }


def filter_gene_by_percentile(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    # return FilterGenesPercentile(min_val=trial.suggest_int(method_name+"min_val",1,10),
    #                              max_val=trial.suggest_int(method_name+"max_val",90,99),
    #                              mode=trial.suggest_categorical(method_name+"mode",["sum","var","cv","rv"]))
    return {
        method_name + "min_val": trial.suggest_int(method_name + "min_val", 1, 10),
        method_name + "max_val": trial.suggest_int(method_name + "max_val", 90, 99),
        method_name + "mode": trial.suggest_categorical(method_name + "mode", ["sum", "var", "cv", "rv"])
    }


def filter_gene_by_count(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    method = trial.suggest_categorical(method_name + "method", ['min_counts', 'min_cells', 'max_counts', 'max_cells'])
    if method == "min_counts":
        num = trial.suggest_int(method_name + "num", 2, 10)
    if method == "min_cells":
        num = trial.suggest_int(method_name + "num", 2, 10)
    if method == "max_counts":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    if method == "max_cells":
        num = trial.suggest_int(method_name + "num", 500, 1000)
    # return AnnDataTransform(sc.pp.filter_genes,method=num)
    return {method_name + "method": method, method_name + "num": num}


def log1p(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    # return AnnDataTransform(sc.pp.log1p,base=trial.suggest_int(method_name+"base",2,10))
    return {method_name + "base": trial.suggest_int(method_name + "base", 2, 10)}


def scTransform(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    return {method_name + "min_cells": trial.suggest_int(method_name + "min_cells", 1, 10)}


    # return ScTransformR(min_cells=trial.suggest_int(method_name+"min_cells",1,10))
def scaleFeature(trial: optuna.Trial):  #eps未优化
    method_name = str(sys._getframe().f_code.co_name) + "_"
    # return ScaleFeature(mode=trial.suggest_categorical(method_name+"mode",["normalize", "standardize", "minmax", "l2"]))
    return {
        method_name + "mode": trial.suggest_categorical(method_name + "mode",
                                                        ["normalize", "standardize", "minmax", "l2"])
    }


def normalize_total(trial: optuna.Trial):
    method_name = str(sys._getframe().f_code.co_name) + "_"
    exclude_highly_expressed = trial.suggest_categorical(method_name + "exclude_highly_expressed", [False, True])

    if exclude_highly_expressed:
        max_fraction = trial.suggest_float(method_name + "max_fraction", 0.04, 0.1)
        return {
            method_name + "exclude_highly_expressed":
            trial.suggest_categorical(method_name + "exclude_highly_expressed", [False, True]),
            method_name + "max_fraction":
            max_fraction
        }
    else:
        return {
            method_name + "exclude_highly_expressed":
            trial.suggest_categorical(method_name + "exclude_highly_expressed", [False, True])
        }
    # return AnnDataTransform(sc.pp.normalize_total,target_sum=trial.suggest_categorical(method_name+"target_sum",[1e4,1e5,1e6]),exclude_highly_expressed=exclude_highly_expressed,max_fraction=max_fraction)


# 从 JSON 文件中读取数据

parameter_config = None


def objective(trial):
    # method_name=str(sys._getframe().f_code.co_name)+"_"
    # exclude_highly_expressed=trial.suggest_categorical(method_name+"exclude_highly_expressed",[False,True])
    # if exclude_highly_expressed:
    #     max_fraction=trial.suggest_float(method_name+"max_fraction",0.04,0.1)
    # print(exclude_highly_expressed)
    # if exclude_highly_expressed:
    #     print(max_fraction)
    global parameter_config
    parameter_config = {}
    for f_str in fun_list:
        fun_i = eval(f_str)
        parameter_config.update(fun_i(trial))
        # transforms.append(fun_i(trial))
    return -1
