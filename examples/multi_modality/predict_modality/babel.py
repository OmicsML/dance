import argparse
import logging
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import skorch.helper
import torch
import torch.nn as nn
import torch.nn.functional as F

import dance.utils.loss as loss_functions
from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.babel import BabelWrapper
from dance.utils import PairedDataset, set_seed

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    OPTIMIZER_DICT = {
        "adam": torch.optim.Adam,
        "rmsprop": torch.optim.RMSprop,
    }
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2_rna')
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)
    parser.add_argument('-m', '--model_folder', default='./models')
    parser.add_argument("--outdir", "-o", default='./logs', help="Directory to output to")
    parser.add_argument(
        "--lossweight",
        type=float,
        default=1.,
        help="Relative loss weight",
    )
    parser.add_argument("--lr", "-l", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batchsize", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimensions")
    parser.add_argument("--earlystop", type=int, default=20, help="Early stopping after N epochs")
    parser.add_argument(
        "--naive",
        "-n",
        action="store_true",
        help="Use a naive model instead of lego model",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=500)
    args = parser.parse_args()
    args.resume = True

    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)
    dataset = ModalityPredictionDataset(args.subtask).load_data().preprocess('feature_selection')
    device = args.device
    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}/training_{args.subtask}_{args.rnd_seed}.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    idx = np.random.permutation(dataset.modalities[0].shape[0])
    train_idx = idx[:int(idx.shape[0] * 0.85)]
    val_idx = idx[int(idx.shape[0] * 0.85):]
    train_mod1 = torch.from_numpy(dataset.numpy_features(0)[train_idx]).float()
    train_mod2 = torch.from_numpy(dataset.numpy_features(1)[train_idx]).float()
    valid_mod1 = torch.from_numpy(dataset.numpy_features(0)[val_idx]).float()
    valid_mod2 = torch.from_numpy(dataset.numpy_features(1)[val_idx]).float()
    test_mod1 = torch.from_numpy(dataset.numpy_features(2)).float()
    test_mod2 = torch.from_numpy(dataset.numpy_features(3)).float()

    sc_dual_train_dataset = PairedDataset(train_mod1, train_mod2)
    sc_dual_valid_dataset = PairedDataset(valid_mod1, valid_mod2)

    model = BabelWrapper(args, dataset)
    model.fit(sc_dual_train_dataset, sc_dual_valid_dataset, args.max_epochs)
    print(model.predict(test_mod1))
    print(model.score(test_mod1, test_mod2))
""" To reproduce BABEL on other samples, please refer to command lines belows:
GEX to ADT:
python babel.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

ADT to GEX:
python babel.py --subtask openproblems_bmmc_cite_phase2_mod2 --device cuda

GEX to ATAC:
python babel.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

ATAC to GEX:
python babel.py --subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda
"""
