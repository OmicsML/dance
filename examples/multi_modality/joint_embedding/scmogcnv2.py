import argparse
import random

import anndata as ad
import dgl
import numpy as np
import scanpy as sc
import torch

import dance.utils.metrics as metrics
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcnv2 import ScMoGCNWrapper
from dance.transforms.graph_construct import construct_basic_feature_graph, gen_batch_features
from dance.utils import set_seed

#
# def scmogcn_test(adata):
#     sc._settings.ScanpyConfig.n_jobs = 4
#     adata_sol = dataset.test_sol
#     adata.obs['batch'] = adata_sol.obs['batch'][adata.obs_names]
#     adata.obs['cell_type'] = adata_sol.obs['cell_type'][adata.obs_names]
#     print(adata.shape,adata_sol.shape)
#     adata_bc = adata.obs_names
#     adata_sol_bc = adata_sol.obs_names
#     select = [item in adata_bc for item in adata_sol_bc]
#     adata_sol = adata_sol[select, :]
#     print(adata.shape, adata_sol.shape)
#
#     adata.obsm['X_emb'] = adata.X
#     nmi = metrics.get_nmi(adata)
#     cell_type_asw = metrics.get_cell_type_ASW(adata)
#     cc_con = metrics.get_cell_cycle_conservation(adata, adata_sol)
#     traj_con = metrics.get_traj_conservation(adata, adata_sol)
#     batch_asw = metrics.get_batch_ASW(adata)
#     graph_score = metrics.get_graph_connectivity(adata)
#
#     print('nmi %.4f, celltype_asw %.4f, cc_con %.4f, traj_con %.4f, batch_asw %.4f, graph_score %.4f\n' % (
#     nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score))
#
#     print('average metric: %.4f' % np.mean(
#         [round(i, 4) for i in [nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score]]))

if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2',
                        choices=['openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2'])
    parser.add_argument('-d', '--data_folder', default='./data/joint_embedding')
    parser.add_argument('-pre', '--pretrained_folder', default='./data/joint_embedding/pretrained')
    parser.add_argument('-csv', '--csv_path', default='decoupled_lsi.csv')
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    parser.add_argument('-prefix', '--prefix', default='dance_openproblems_bmmc_atac2rna_test')

    parser.add_argument('-pww', '--pathway_weight', default='pearson', choices=['cos', 'one', 'pearson'])
    parser.add_argument('-pwth', '--pathway_threshold', type=float, default=-1.0)
    parser.add_argument('-l', '--log_folder', default='../../../single_cell/logs')
    parser.add_argument('-m', '--model_folder', default='../../../single_cell/models')
    parser.add_argument('-r', '--result_folder', default='../../../single_cell/results')
    parser.add_argument('-e', '--epoch', type=int, default=1000)
    parser.add_argument('-nbf', '--no_batch_features', action='store_true')
    parser.add_argument('-npw', '--no_pathway', action='store_true')
    parser.add_argument('-opw', '--only_pathway', action='store_true')
    parser.add_argument('-res', '--residual', default='res_cat', choices=['none', 'res_add', 'res_cat'])
    parser.add_argument('-inres', '--initial_residual', action='store_true')
    parser.add_argument('-pwagg', '--pathway_aggregation', default='alpha',
                        choices=['sum', 'attention', 'two_gate', 'one_gate', 'alpha', 'cat'])
    parser.add_argument('-pwalpha', '--pathway_alpha', type=float, default=0.5)
    parser.add_argument('-nrc', '--no_readout_concatenate', action='store_true')
    parser.add_argument('-nm', '--normalization', default='group', choices=['batch', 'layer', 'group', 'none'])
    parser.add_argument('-ac', '--activation', default='gelu', choices=['leaky_relu', 'relu', 'prelu', 'gelu'])
    parser.add_argument('-em', '--embedding_layers', default=1, type=int, choices=[1, 2, 3])
    parser.add_argument('-ro', '--readout_layers', default=2, type=int, choices=[1, 2, 3])
    parser.add_argument('-conv', '--conv_layers', default=3, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('-agg', '--agg_function', default='mean', choices=['gcn', 'mean'])
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-sb', '--save_best', action='store_true')
    parser.add_argument('-sf', '--save_final', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.99)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3)
    parser.add_argument('-hid', '--hidden_size', type=int, default=48)
    parser.add_argument('-edd', '--edge_dropout', type=float, default=0)
    parser.add_argument('-mdd', '--model_dropout', type=float, default=0.4)
    parser.add_argument('-es', '--early_stopping', type=int, default=20)
    parser.add_argument('-or', '--output_relu', default='none', choices=['relu', 'leaky_relu', 'none'])
    parser.add_argument('-i', '--inductive', default='trans', choices=['normal', 'opt', 'trans'])
    parser.add_argument('-sa', '--subpath_activation', action='store_true')
    parser.add_argument('-ci', '--cell_init', default='none', choices=['none', 'pca'])
    parser.add_argument('-bas', '--batch_seperation', action='store_true')
    parser.add_argument('-pwpath', '--pathway_path', default='../../../single_cell/task1/h.all.v7.4')
    parser.add_argument('-ws', '--weighted_sum', action='store_true')

    args = parser.parse_args()
    args.rnd_seed = 1507659161

    device = args.device
    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)

    args.no_pathway = True
    args.no_batch_features = True  #False
    args.weighted_sum = True
    args.hidden_size = 56


    dataset = JointEmbeddingNIPSDataset(args.subtask, data_dir=args.data_folder).load_data()\
        .load_metadata().load_sol().preprocess('aux', args.pretrained_folder)
    X = torch.cat([dataset.tensor_features(0), dataset.tensor_features(1)], 1)
    Y_train = dataset.preprocessed_data['Y_train']
    # args.hidden_size = Y_train[0].max()+Y_train[1].max()+2+10
    print(Y_train[0].max() + 1, Y_train[0].max() + Y_train[1].max() + 2)

    g = construct_basic_feature_graph(X, device=device)
    args.FEATURE_SIZE = X.shape[1]
    X = X.to(device)
    args.feat1 = dataset.modalities[0].shape[1]
    args.feat2 = dataset.modalities[1].shape[1]

    if not args.no_batch_features:
        batch_features = gen_batch_features([dataset.modalities[0]]).to(device)
        args.BATCH_NUM = batch_features.shape[1]
        g.nodes['cell'].data['bf'] = batch_features

    model = ScMoGCNWrapper(args)

    model.load(f'models/model_joint_embedding_{rndseed}.pth')
    # model.fit(g, X, Y_train, args.epoch)
    # model.load(f'models/model_joint_embedding_{rndseed}.pth')

    with torch.no_grad():
        embeds = model.predict(g).cpu().numpy()
        print(embeds)

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
    adata.write_h5ad(f'./joint_embedding_{rndseed}.h5ad', compression="gzip")

    # scmvae test
    metrics.labeled_clustering_evaluate(adata, dataset)  #, 45)
