import argparse
import math
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

import dance.utils.metrics as metrics
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
# from scMVAE.utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, \
#     load_checkpoint, adjust_learning_rate
from dance.modules.multi_modality.joint_embedding.scmvae import scMVAE
from dance.transforms.preprocess import calculate_log_library_size


def parameter_setting():
    parser = argparse.ArgumentParser(description='Single cell Multi-omics data analysis')

    outPath = './new_test/'

    # parser.add_argument ('--latent_fusion', '-olf1', type=str, default='First_simulate_fusion.csv',
    #                     help='fusion latent code file')
    # parser.add_argument('--latent_1', '-ol1', type=str, default='scRNA_latent_combine.csv',
    #                     help='first latent code file')
    # parser.add_argument('--latent_2', '-ol2', type=str, default='scATAC_latent.csv', help='seconde latent code file')
    # parser.add_argument('--denoised_1', '-od1', type=str, default='scRNA_seq_denoised.csv',
    #                     help='outfile for denoised file1')
    # parser.add_argument('--normalized_1', '-on1', type=str, default='scRNA_seq_normalized_combine.tsv',
    #                     help='outfile for normalized file1')
    # parser.add_argument('--denoised_2', '-od2', type=str, default='scATAC_seq_denoised.csv',
    #                     help='outfile for denoised file2')

    parser.add_argument('--workdir', '-wk', type=str, default=outPath, help='work path')
    parser.add_argument('--outdir', '-od', type=str, default=outPath, help='Output path')

    parser.add_argument('--lr', type=float, default=1E-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default=0.01, help='eps')

    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')
    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--latent', '-l', type=int, default=10, help='latent layer dim')
    parser.add_argument('--max_epoch', '-me', type=int, default=25, help='Max epoches')
    parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
    parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default=5,
                        help='Epoch per test, must smaller than max iteration.')
    parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2')
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('--final_rate', type=float, default=1e-4)
    parser.add_argument('--scale_factor', type=float, default=4)

    return parser


if __name__ == "__main__":
    parser = parameter_setting()
    args = parser.parse_args()
    assert args.max_iteration > args.epoch_per_test

    dataset = JointEmbeddingNIPSDataset(args.subtask, data_dir='./data/joint_embedding').load_data() \
        .load_metadata().load_sol().preprocess(kind='feature_selection')

    adata1 = dataset.modalities[0]
    adata2 = dataset.modalities[1]
    adata1.X = adata1.layers['counts']
    adata2.X = adata2.layers['counts']

    idx = np.random.permutation(adata1.shape[0])
    train_index = idx[:int(len(idx) * 0.85)]
    valid_index = idx[int(len(idx) * 0.85):]

    lib_mean1, lib_var1 = calculate_log_library_size(adata1.X)
    lib_mean2, lib_var2 = calculate_log_library_size(adata2.X)

    Nfeature1 = np.shape(adata1.X)[1]
    Nfeature2 = np.shape(adata2.X)[1]

    device = torch.device(args.device)

    model = scMVAE(
        encoder_1=[Nfeature1, 1024, 128, 128],
        hidden_1=128,
        Z_DIMS=22,
        decoder_share=[22, 128, 256],
        share_hidden=128,
        decoder_1=[128, 128, 1024],
        hidden_2=1024,
        encoder_l=[Nfeature1, 128],
        hidden3=128,
        encoder_2=[Nfeature2, 1024, 128, 128],
        hidden_4=128,
        encoder_l1=[Nfeature2, 128],
        hidden3_1=128,
        decoder_2=[128, 128, 1024],
        hidden_5=1024,
        drop_rate=0.1,
        log_variational=True,
        Type="ZINB",
        device=device,
        n_centroids=22,
        penality="GMM",
        model=1,
    )

    args.lr = 0.001
    args.anneal_epoch = 200

    model.to(device)
    train = data_utils.TensorDataset(torch.from_numpy(adata1.X[train_index].todense()),
                                     torch.from_numpy(lib_mean1[train_index]), torch.from_numpy(lib_var1[train_index]),
                                     torch.from_numpy(lib_mean2[train_index]), torch.from_numpy(lib_var2[train_index]),
                                     torch.from_numpy(adata2.X[train_index].todense()))

    valid = data_utils.TensorDataset(torch.from_numpy(adata1.X[valid_index].todense()),
                                     torch.from_numpy(lib_mean1[valid_index]), torch.from_numpy(lib_var1[valid_index]),
                                     torch.from_numpy(lib_mean2[valid_index]), torch.from_numpy(lib_var2[valid_index]),
                                     torch.from_numpy(adata2.X[valid_index].todense()))

    total = data_utils.TensorDataset(torch.from_numpy(adata1.X.todense()), torch.from_numpy(adata2.X.todense()))

    total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)
    model.init_gmm_params(total_loader)
    model.fit(args, train, valid, args.final_rate, args.scale_factor, device)

    # model.load_state_dict(torch.load('./saved_model/model_best.pth.tar') )

    test_mod1 = dataset.tensor_features(0).to(device)
    test_mod2 = dataset.tensor_features(1).to(device)
    embeds = model.predict(test_mod1, test_mod2).cpu().numpy()
    print(embeds)
    print(model.score(test_mod1, test_mod2, dataset.test_sol))

    # mod1_obs = dataset.modalities[0].obs
    # mod1_uns = dataset.modalities[0].uns
    # adata = ad.AnnData(
    #     X=embeds,
    #     obs=mod1_obs,
    #     uns={
    #         'dataset_id': mod1_uns['dataset_id'],
    #         'method_id': 'scmogcn',
    #     },
    # )
    # # adata.write_h5ad(f'./joint_embedding_scmvae.h5ad', compression="gzip")
    #
    # metrics.labeled_clustering_evaluate(adata, dataset)
""" To reproduce scMVAE on other samples, please refer to command lines belows:
GEX-ADT:
python scmvae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
python scmvae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
