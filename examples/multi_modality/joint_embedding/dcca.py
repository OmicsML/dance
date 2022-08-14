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
from dance.modules.multi_modality.joint_embedding.dcca import DCCA
# from scMVAE.utilities import read_dataset, normalize, calculate_log_library_size, parameter_setting, save_checkpoint, \
#     load_checkpoint, adjust_learning_rate
from dance.modules.multi_modality.joint_embedding.scmvae import scMVAE
from dance.transforms.preprocess import calculate_log_library_size


def parameter_setting():
    parser = argparse.ArgumentParser(description='Single cell Multi-omics data analysis')

    outPath = './new_test/'

    # parser.add_argument('--File1', '-F1', type=str, default='5-counts-RNA.tsv', help='input file name1')
    # parser.add_argument('--File2', '-F2', type=str, default='0.75-0.2-counts-ATAC.tsv', help='input file name2')
    # parser.add_argument('--File2_1', '-F2_1', type=str, default='0.75-0.2-counts-ATAC_binary.tsv',
    #                     help='input file name2_1')
    #
    # parser.add_argument('--File3', '-F3', type=str, default='5-cellinfo-RNA.tsv', help='input meta file')
    # parser.add_argument('--File_combine', '-F_com', type=str, default='Gene_chromatin_order_combine.tsv',
    #                     help='input combine file name')
    # parser.add_argument('--File_mofa', '-F_mofa', type=str, default='MOFA_combine_cluster.csv',
    #                     help='cluster for mofa predicted')

    parser.add_argument('--latent_fusion', '-olf1', type=str, default='First_simulate_fusion.csv',
                        help='fusion latent code file')
    parser.add_argument('--latent_1', '-ol1', type=str, default='scRNA_latent_combine.csv',
                        help='first latent code file')
    parser.add_argument('--latent_2', '-ol2', type=str, default='scATAC_latent.csv', help='seconde latent code file')
    parser.add_argument('--denoised_1', '-od1', type=str, default='scRNA_seq_denoised.csv',
                        help='outfile for denoised file1')
    parser.add_argument('--normalized_1', '-on1', type=str, default='scRNA_seq_normalized_combine.tsv',
                        help='outfile for normalized file1')
    parser.add_argument('--denoised_2', '-od2', type=str, default='scATAC_seq_denoised.csv',
                        help='outfile for denoised file2')

    parser.add_argument('--workdir', '-wk', type=str, default=outPath, help='work path')
    parser.add_argument('--outdir', '-od', type=str, default=outPath, help='Output path')

    parser.add_argument('--lr', type=float, default=1E-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default=0.01, help='eps')

    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size')

    parser.add_argument('--seed', type=int, default=200, help='Random seed for repeat results')
    parser.add_argument('--latent', '-l', type=int, default=10, help='latent layer dim')
    parser.add_argument('--max_epoch', '-me', type=int, default=100, help='Max epoches')
    parser.add_argument('--max_iteration', '-mi', type=int, default=3000, help='Max iteration')
    parser.add_argument('--anneal_epoch', '-ae', type=int, default=200, help='Anneal epoch')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default=5, help='Epoch per test')
    parser.add_argument('--max_ARI', '-ma', type=int, default=-200, help='initial ARI')
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2')
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('--final_rate', type=float, default=1e-4)
    parser.add_argument('--scale_factor', type=float, default=4)

    return parser


if __name__ == "__main__":
    parser = parameter_setting()
    args = parser.parse_args()

    args.sf1 = 5
    args.sf2 = 1
    args.cluster1 = args.cluster2 = 4
    args.lr1 = 0.01
    args.flr1 = 0.001
    args.lr2 = 0.005
    args.flr2 = 0.0005

    dataset = JointEmbeddingNIPSDataset(args.subtask, data_dir='./data/joint_embedding').load_data() \
        .load_metadata().load_sol().preprocess('feature_selection')

    adata = dataset.modalities[0]
    adata1 = dataset.modalities[1]
    idx = np.random.permutation(adata.shape[0])
    train_index = idx[:int(len(idx) * 0.85)]
    test_index = idx[int(len(idx) * 0.85):]
    label_ground_truth = adata.obs['batch']

    Nfeature1 = np.shape(adata.X)[1]
    Nfeature2 = np.shape(adata1.X)[1]

    device = torch.device(args.device)

    model = DCCA(layer_e_1=[Nfeature1, 128], hidden1_1=128, Zdim_1=4, layer_d_1=[4, 128], hidden2_1=128,
                 layer_e_2=[Nfeature2, 1500, 128], hidden1_2=128, Zdim_2=4, layer_d_2=[4], hidden2_2=4, args=args,
                 ground_truth1=label_ground_truth, ground_truth2=label_ground_truth, Type_1="NB", Type_2="Bernoulli",
                 cycle=1, attention_loss="Eucli").to(device)

    model.to(device)
    train = data_utils.TensorDataset(torch.from_numpy(adata[train_index].X.todense()),
                                     torch.from_numpy(adata.layers['counts'][train_index].todense()),
                                     torch.from_numpy(adata.obs['size_factors'][train_index].values),
                                     torch.from_numpy(adata1[train_index].X.todense()),
                                     torch.from_numpy(adata1.layers['counts'][train_index].todense()),
                                     torch.from_numpy(adata.obs['size_factors'][train_index].values))

    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    test = data_utils.TensorDataset(torch.from_numpy(adata[test_index].X.todense()),
                                    torch.from_numpy(adata.layers['counts'][test_index].todense()),
                                    torch.from_numpy(adata.obs['size_factors'][test_index].values),
                                    torch.from_numpy(adata1[test_index].X.todense()),
                                    torch.from_numpy(adata1.layers['counts'][test_index].todense()),
                                    torch.from_numpy(adata.obs['size_factors'][test_index].values))

    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False)

    total = data_utils.TensorDataset(torch.from_numpy(adata.X.todense()),
                                     torch.from_numpy(adata.layers['counts'].todense()),
                                     torch.from_numpy(adata.obs['size_factors'].values),
                                     torch.from_numpy(adata1.X.todense()),
                                     torch.from_numpy(adata1.layers['counts'].todense()),
                                     torch.from_numpy(adata.obs['size_factors'].values))
    total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)

    model.fit(train_loader, test_loader, total_loader, "RNA")

    with torch.no_grad():
        emb1, emb2 = model.predict(total_loader)

    embeds = np.concatenate([emb1, emb2], 1)
    print(embeds)
    print(model.score(total_loader))

    mod1_obs = dataset.modalities[0].obs
    mod1_uns = dataset.modalities[0].uns
    adata = ad.AnnData(
        X=embeds,
        obs=mod1_obs,
        uns={
            'dataset_id': mod1_uns['dataset_id'],
            'method_id': 'scmogcn',
        },
    )
    # adata.write_h5ad(f'./joint_embedding_scmvae.h5ad', compression="gzip")

    metrics.labeled_clustering_evaluate(adata, dataset)
""" To reproduce DCCA on other samples, please refer to command lines belows:
GEX-ADT:
python dcca.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
python dcca.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
