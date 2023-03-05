import argparse
import random

import anndata as ad
import dgl
import mudata
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix

import dance.utils.metrics as metrics
from dance.data import Data
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcn import ScMoGCNWrapper
from dance.transforms.graph.cell_feature_graph import CellFeatureBipartiteGraph
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
    mod1 = dataset.modalities[0]
    mod2 = dataset.modalities[1]
    mod1.var_names_make_unique()
    mod2.var_names_make_unique()
    mod1.obs_names_make_unique()
    mod2.obs_names = mod1.obs_names
    mdata = mudata.MuData({"mod1": mod1, "mod2": mod2})
    mdata.var_names_make_unique()
    # train_size = int(mod1.shape[0] * 0.85)
    train_size = dataset.train_size
    data = Data(mdata)
    data = CellFeatureBipartiteGraph(cell_feature_channel='X_pca', mod='mod1')(data)
    data = CellFeatureBipartiteGraph(cell_feature_channel='X_pca', mod='mod2')(data)
    data.set_config(feature_mod=["mod1", "mod2"], label_mod=["mod1", "mod1", "mod1", "mod1",
                                                             "mod1"], feature_channel=['X_pca', 'X_pca'],
                    label_channel=['cell_type', 'batch_label', 'phase_labels', 'S_scores', 'G2M_scores'])
    (x_mod1, x_mod2), (cell_type, batch_label, phase_label, S_score, G2M_score) = data.get_data(return_type='torch')
    phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)

    model = ScMoGCNWrapper(args, num_celL_types=int(cell_type.max() + 1), num_batches=int(batch_label.max() + 1),
                           num_phases=phase_score.shape[1], num_features=x_mod1.shape[1] + x_mod2.shape[1])
    model.fit(
        g_mod1=data.data['mod1'].uns['g'],
        g_mod2=data.data['mod2'].uns['g'],
        train_size=train_size,
        cell_type=cell_type,
        batch_label=batch_label,
        phase_score=phase_score,
    )
    model.load(f'models/model_joint_embedding_{rndseed}.pth')

    with torch.no_grad():
        test_id = np.arange(train_size, x_mod1.shape[0])
        # test_id = np.arange(x_mod1.shape[0])
        labels = cell_type.numpy()[test_id]
        embeds = model.predict(test_id).cpu().numpy()
        print(embeds)
        print(model.score(test_id, labels, metric='clustering'))

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
