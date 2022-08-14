import argparse
import random

import numpy as np
import torch

from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.scmm import MMVAE
from dance.utils import set_seed

if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./predict_modality/output', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2_rna')
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)

    parser.add_argument('--experiment', type=str, default='test', metavar='E', help='experiment name')
    parser.add_argument('--obj', type=str, default='m_elbo_naive_warmup', metavar='O',
                        help='objective to use (default: elbo)')
    parser.add_argument(
        '--llik_scaling', type=float, default=1., help='likelihood scaling for cub images/svhn modality when running in'
        'multimodal setting, set as 0 to use default value')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data (default: 256)')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='L', help='learning rate (default: 1e-3)')
    parser.add_argument('--latent_dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 20)')
    parser.add_argument('--num_hidden_layers', type=int, default=2, metavar='H',
                        help='number of hidden layers in enc and dec (default: 2)')
    parser.add_argument('--r_hidden_dim', type=int, default=100, help='number of hidden units in enc/dec for gene')
    parser.add_argument('--p_hidden_dim', type=int, default=20,
                        help='number of hidden units in enc/dec for protein/peak')
    parser.add_argument('--pre_trained', type=str, default="",
                        help='path to pre-trained model (train from scratch if empty)')
    parser.add_argument('--learn_prior', type=bool, default=True, help='learn model prior parameters')
    parser.add_argument('--print_freq', type=int, default=0, metavar='f',
                        help='frequency with which to print stats (default: 0)')
    parser.add_argument('--deterministic_warmup', type=int, default=50, metavar='W', help='deterministic warmup')
    args = parser.parse_args()

    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)
    dataset = ModalityPredictionDataset(args.subtask).load_data().preprocess('feature_selection')
    device = args.device
    subtask = args.subtask

    args.r_dim = dataset.modalities[0].shape[1]
    args.p_dim = dataset.modalities[1].shape[1]

    model_class = 'rna-protein' if subtask == 'openproblems_bmmc_cite_phase2_rna' else 'rna-dna'
    model = MMVAE(model_class, args).to(device)

    model.fit(dataset)
    test_X = torch.from_numpy(dataset.numpy_features(2, True)).to(device).float()
    test_Y = torch.from_numpy(dataset.numpy_features(3, True)).to(device).float()
    print(model.predict(test_X))
    print(model.score(test_X, test_Y))
""" To reproduce scMM on other samples, please refer to command lines belows:
GEX to ADT:
python scmm.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

ADT to GEX:
python scmm.py --subtask openproblems_bmmc_cite_phase2_mod2 --device cuda

GEX to ATAC:
python scmm.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

ATAC to GEX:
python scmm.py --subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda
"""
