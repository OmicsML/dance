import argparse
import random

import anndata as ad
import dgl
import numpy as np
import scanpy as sc
import torch

import dance.utils.metrics as metrics
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcn import ScMoGCNWrapper
from dance.transforms.graph_construct import basic_feature_graph_propagation, construct_basic_feature_graph
from dance.utils import set_seed

if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2',
                        choices=['openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2'])
    parser.add_argument('-d', '--data_folder', default='./data/joint_embedding')
    parser.add_argument('-pre', '--pretrained_folder', default='./data/joint_embedding/pretrained')
    parser.add_argument('-csv', '--csv_path', default='decoupled_lsi.csv')
    parser.add_argument('-l', '--layers', default=3, type=int, choices=[3, 4, 5, 6, 7])
    parser.add_argument('-dis', '--disable_propagation', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    parser.add_argument('-nm', '--normalize', default=1, type=int, choices=[0, 1])

    args = parser.parse_args()

    device = args.device
    pre_normalize = bool(args.normalize)
    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)

    dataset = JointEmbeddingNIPSDataset(args.subtask, data_dir=args.data_folder).load_data()\
        .load_metadata().load_sol().preprocess('aux', args.pretrained_folder).normalize()
    X_train, Y_train, X_test = dataset.preprocessed_data['X_train'], dataset.preprocessed_data['Y_train'], \
                                       dataset.preprocessed_data['X_test']

    g = construct_basic_feature_graph(X_train, X_test, device=device)
    X = basic_feature_graph_propagation(g, layers=args.layers, device=device)

    l = args.layers - 1

    model = ScMoGCNWrapper(args, dataset)
    model.fit(dataset, X, Y_train)
    model.load(f'models/model_joint_embedding_{rndseed}.pth')

    with torch.no_grad():
        test_id = np.arange(X[0].shape[0])
        labels = dataset.test_sol.obs['cell_type'].to_numpy()
        embeds = model.predict(X, test_id).cpu().numpy()
        print(embeds)
        print(model.score(X, test_id, labels, 'clustering'))

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
    # adata.write_h5ad(f'./joint_embedding_{rndseed}.h5ad', compression="gzip")
    #
    # # scmvae test
    # metrics.labeled_clustering_evaluate(adata, dataset)
""" To reproduce scMoGCN on other samples, please refer to command lines belows:
GEX-ADT:
python scmogcn.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
python scmogcn.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
