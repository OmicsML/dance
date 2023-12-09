import argparse
import os
from argparse import Namespace

import anndata
import mudata
import numpy as np
import pandas as pd
import torch

from dance.data import Data
from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.scmogcn import ScMoGCNWrapper
from dance.transforms.cell_feature import BatchFeature
from dance.transforms.graph import ScMoGNNGraph
from dance.utils import set_seed


def pipeline(inductive=False, verbose=2, logger=None, **kwargs):
    PREFIX = kwargs["prefix"]
    os.makedirs(kwargs["log_folder"], exist_ok=True)
    os.makedirs(kwargs["model_folder"], exist_ok=True)
    os.makedirs(kwargs["result_folder"], exist_ok=True)

    subtask = kwargs["subtask"]
    dataset = ModalityPredictionDataset(subtask, preprocess=kwargs["preprocessing"])
    dataset.download_pathway()
    modalities = dataset.load_raw_data()

    mod1 = anndata.concat((modalities[0], modalities[2]))
    mod2 = anndata.concat((modalities[1], modalities[3]))
    mod1.var_names_make_unique()
    mod2.var_names_make_unique()

    mod1_batch_cat, mod2_batch_cat = map(lambda x: x.obs["batch"].astype("category"), (mod1, mod2))
    mod1_batch, mod2_batch = map(lambda x: x if "unknown" in x.cat.categories else x.cat.add_categories("unknown"),
                                 (mod1_batch_cat, mod2_batch_cat))

    mod1.obs["batch"] = mod1_batch.fillna("unknown")
    mod2.obs["batch"] = mod2_batch.fillna("unknown")

    mdata = mudata.MuData({"mod1": mod1, "mod2": mod2})
    train_size = modalities[0].shape[0]
    data = Data(mdata, train_size=train_size)
    data.set_config(feature_mod="mod1", label_mod="mod2")

    data = ScMoGNNGraph(inductive, kwargs["cell_init"], kwargs["pathway"], kwargs["subtask"], kwargs["pathway_weight"],
                        kwargs["pathway_threshold"], kwargs["pathway_path"])(data)
    if not kwargs["no_batch_features"]:
        data = BatchFeature()(data)

    idx = np.random.permutation(modalities[0].shape[0])
    split = {"train": idx[:-int(len(idx) * 0.15)], "valid": idx[-int(len(idx) * 0.15):]}
    kwargs["FEATURE_SIZE"] = modalities[0].shape[1]
    kwargs["TRAIN_SIZE"] = modalities[0].shape[0]
    kwargs["OUTPUT_SIZE"] = modalities[1].shape[1]
    kwargs["CELL_SIZE"] = modalities[0].shape[0] + modalities[2].shape[0]

    if inductive:
        g, gtest = data.data.uns["g"], data.data.uns["gtest"]
    else:
        gtest = g = data.data.uns["g"]

    _, y_train = data.get_train_data(return_type="torch")
    _, y_test = data.get_test_data(return_type="torch")

    if not kwargs["no_batch_features"]:
        batch_features = torch.from_numpy(data.data["mod1"].obsm["batch_features"]).float()
        kwargs["BATCH_NUM"] = batch_features.shape[1]
        if inductive:
            g.nodes["cell"].data["bf"] = batch_features[:kwargs["TRAIN_SIZE"]]
            gtest.nodes["cell"].data["bf"] = batch_features
        else:
            g.nodes["cell"].data["bf"] = batch_features

    res = pd.DataFrame({'rmse': [], 'seed': [], 'subtask': [], 'method': []})
    for k in range(args.runs):
        set_seed(args.seed)
        model = ScMoGCNWrapper(Namespace(**kwargs))
        if verbose > 1:
            logger = open(f"{kwargs['log_folder']}/{PREFIX}.log", "w")
            logger.write(str(kwargs) + "\n")

        if kwargs["sampling"]:
            model.fit_with_sampling(g, y_train, split, not inductive, verbose, y_test, logger)
        else:
            model.fit(g, y_train, split, not inductive, verbose, y_test, logger)

        print(model.predict(g, np.arange(kwargs["TRAIN_SIZE"], kwargs["CELL_SIZE"]), device="cpu"))
        res = res.append(
            {
                'rmse': model.score(g, np.arange(kwargs["TRAIN_SIZE"], kwargs["CELL_SIZE"]), y_test, device="cpu"),
                'seed': args.seed,
                'subtask': args.subtask,
                'method': 'scmogcn',
            }, ignore_index=True)
        args.seed = args.seed + 1
    print(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-prefix", "--prefix", default="dance_openproblems_bmmc_atac2rna_test")
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("-pww", "--pathway_weight", default="pearson", choices=["cos", "one", "pearson"])
    parser.add_argument("-pwth", "--pathway_threshold", type=float, default=-1.0)
    parser.add_argument("-l", "--log_folder", default="./logs")
    parser.add_argument("-m", "--model_folder", default="./models")
    parser.add_argument("-r", "--result_folder", default="./results")
    parser.add_argument("-e", "--epoch", type=int, default=15000)
    parser.add_argument("-nbf", "--no_batch_features", action="store_true")
    parser.add_argument("-npw", "--pathway", action="store_true")
    parser.add_argument("-res", "--residual", default="res_cat", choices=["none", "res_add", "res_cat"])
    parser.add_argument("-inres", "--initial_residual", action="store_true")
    parser.add_argument("-pwagg", "--pathway_aggregation", default="alpha",
                        choices=["sum", "attention", "two_gate", "one_gate", "alpha", "cat"])
    parser.add_argument("-pwalpha", "--pathway_alpha", type=float, default=0.25)
    parser.add_argument("-nrc", "--no_readout_concatenate", action="store_true")
    parser.add_argument("-bs", "--batch_size", default=1000, type=int)
    parser.add_argument("-nm", "--normalization", default="group", choices=["batch", "layer", "group", "none"])
    parser.add_argument("-ac", "--activation", default="gelu", choices=["leaky_relu", "relu", "prelu", "gelu"])
    parser.add_argument("-em", "--embedding_layers", default=1, type=int, choices=[1, 2, 3])
    parser.add_argument("-ro", "--readout_layers", default=1, type=int, choices=[1, 2])
    parser.add_argument("-conv", "--conv_layers", default=4, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("-agg", "--agg_function", default="mean", choices=["gcn", "mean"])
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-sb", "--save_best", action="store_true")
    parser.add_argument("-sf", "--save_final", action="store_true")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-2)
    parser.add_argument("-lrd", "--lr_decay", type=float, default=0.99)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-hid", "--hidden_size", type=int, default=48)
    parser.add_argument("-edd", "--edge_dropout", type=float, default=0.3)
    parser.add_argument("-mdd", "--model_dropout", type=float, default=0.2)
    parser.add_argument("-es", "--early_stopping", type=int, default=200)
    parser.add_argument("-c", "--cpu", type=int, default=1)
    parser.add_argument("-or", "--output_relu", default="none", choices=["relu", "leaky_relu", "none"])
    parser.add_argument("-i", "--inductive", action="store_true")
    parser.add_argument("-sa", "--subpath_activation", action="store_true")
    parser.add_argument("-ci", "--cell_init", default="none", choices=["none", "svd"])
    parser.add_argument("-bas", "--batch_seperation", action="store_true")
    parser.add_argument("-pwpath", "--pathway_path", default="./data/h.all.v7.4")
    parser.add_argument("-seed", "--seed", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("-ws", "--weighted_sum", action="store_true")
    parser.add_argument("-samp", "--sampling", action="store_true")
    parser.add_argument("-ns", "--node_sampling_rate", type=float, default=0.5)
    parser.add_argument("-prep", "--preprocessing", default="none", choices=["none", "feature_selection", "svd"])
    parser.add_argument("-lm", "--low_memory", type=bool, default=False)

    args = parser.parse_args()

    # For test only (low gpu memory setting; to reproduce competition result need >20G GPU memory - v100)
    if args.low_memory:
        print("WARNING: Running in low memory mode, some cli settings maybe overwritten!")
        args.preprocessing = "feature_selection"
        args.pathway = False
        args.sampling = True
        args.batch_size = 10000
        args.epoch = 10

    # Regular settings
    if args.subtask in ["openproblems_bmmc_cite_phase2_mod2", "openproblems_bmmc_multiome_phase2_mod2"]:
        args.preprocessing = "feature_selection"
    elif args.subtask in ["openproblems_bmmc_multiome_phase2_rna"]:
        args.preprocessing = 'svd'
    if args.subtask.find("rna") == -1:
        args.pathway = False
    if args.sampling:
        args.pathway = False

    torch.set_num_threads(args.cpu)

    pipeline(**vars(args))
"""To reproduce scMoGCN on other samples, please refer to command lines belows:

GEX to ADT (subset):
$ python scmogcn.py --subtask oopenproblems_bmmc_cite_phase2_rna_subset --device cuda

GEX to ADT:
$ python scmogcn.py --subtask oopenproblems_bmmc_cite_phase2_rna --device cuda -inres -sb -hid=256 -wd 1e-4 -pww 'cos' -es 200 -pwth 0.1 -ws -edd 0.4 -mdd 0.3

ADT to GEX:
$ python scmogcn.py --subtask openproblems_bmmc_cite_phase2_mod2 --device cuda -es 300

GEX to ATAC:
$ python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda -es 300

ATAC to GEX:
$ python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda -es 1000 -e 3000 -edd 0

"""
