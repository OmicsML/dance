import argparse
import random

import anndata
import mudata
import numpy as np
import torch
import torch.nn.functional as F

from dance.data import Data
from dance.datasets.multimodality import ModalityMatchingDataset
from dance.modules.multi_modality.match_modality.scmogcn import ScMoGCNWrapper
from dance.transforms.graph.cell_feature_graph import CellFeatureBipartiteGraph
from dance.utils import set_seed


def normalize(X):
    return (X - X.mean()) / X.std()


if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2_rna',
                        choices=['openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_multiome_phase2_rna'])
    parser.add_argument('-d', '--data_folder', default='./data/modality_matching')
    parser.add_argument('-csv', '--csv_path', default='decoupled_lsi.csv')
    parser.add_argument('-l', '--layers', default=4, type=int, choices=[3, 4, 5, 6, 7])
    parser.add_argument('-lr', '--learning_rate', default=6e-4, type=float)
    parser.add_argument('-dis', '--disable_propagation', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('-aux', '--auxiliary_loss', default=True, type=bool)
    parser.add_argument('-pk', '--pickle_suffix', default='_lsi_input_pca_count.pkl')
    parser.add_argument('-seed', '--rnd_seed', default=rndseed, type=int)
    parser.add_argument('-cpu', '--cpus', default=1, type=int)
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-e', '--epochs', default=2000, type=int)

    args = parser.parse_args()

    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)

    subtask = args.subtask
    data_folder = args.data_folder
    device = args.device
    layers = args.layers
    pkl_path = subtask + args.pickle_suffix

    # DataLoader
    dataset = ModalityMatchingDataset(subtask, data_dir=data_folder).load_data().load_sol().preprocess('pca', pkl_path)
    mod1 = anndata.concat((dataset.modalities[0], dataset.modalities[2]))
    mod2 = anndata.concat((dataset.modalities[1], dataset.modalities[3]))
    train_size = dataset.modalities[0].shape[0]
    mod1.obsm['labels'] = np.concatenate([np.zeros(train_size), np.argmax(dataset.test_sol.X.toarray(), 1)])
    mod1.var_names_make_unique()
    mod2.var_names_make_unique()
    mod1.obs_names_make_unique()
    mod2.obs_names = mod1.obs_names
    mdata = mudata.MuData({"mod1": mod1, "mod2": mod2})
    mdata.var_names_make_unique()
    data = Data(mdata)
    data = CellFeatureBipartiteGraph(cell_feature_channel='X_pca', mod='mod1')(data)
    data = CellFeatureBipartiteGraph(cell_feature_channel='X_pca', mod='mod2')(data)

    # data.set_config(feature_mod=["mod1", "mod2"], label_mod="mod1", feature_channel_type=["obsm", "obsm"],
    #                 feature_channel=["prop", "prop"], label_channel='labels')
    # (x, y), z = data.get_feature(return_type="torch")

    data.set_config(feature_mod=["mod1", "mod2", "mod1", "mod2"], feature_channel_type=["uns", "uns", "obs", "obs"],
                    feature_channel=["g", "g", "batch", "batch"], label_mod="mod1", label_channel='labels')
    (g_mod1, g_mod2, batch_mod1, batch_mod2), z = data.get_data(return_type="default")

    if subtask == 'openproblems_bmmc_cite_phase2_rna':
        HIDDEN_SIZE = 64
        TEMPERATURE = 2.739896
        model = ScMoGCNWrapper(
            args, [[(g_mod1.num_nodes('feature'), 512, 0.25), (512, 512, 0.25),
                    (512, HIDDEN_SIZE)], [(g_mod2.num_nodes('feature'), 512, 0.2), (512, 512, 0.2), (512, HIDDEN_SIZE)],
                   [(HIDDEN_SIZE, 512, 0.2),
                    (512, g_mod1.num_nodes('feature'))], [(HIDDEN_SIZE, 512, 0.2),
                                                          (512, g_mod2.num_nodes('feature'))]], TEMPERATURE)
    else:
        HIDDEN_SIZE = 256
        TEMPERATURE = 3.065016
        model = ScMoGCNWrapper(
            args, [[(g_mod1.num_nodes('feature'), 1024, 0.5), (1024, 1024, 0.5),
                    (1024, HIDDEN_SIZE)], [(g_mod2.num_nodes('feature'), 2048, 0.5), (2048, HIDDEN_SIZE)],
                   [(HIDDEN_SIZE, 512, 0.2),
                    (512, g_mod1.num_nodes('feature'))], [(HIDDEN_SIZE, 512, 0.2),
                                                          (512, g_mod2.num_nodes('feature'))]], TEMPERATURE)

    train_size = dataset.sparse_features()[0].shape[0]
    z_test = F.one_hot(torch.from_numpy(z[train_size:]).long())
    labels1 = torch.argmax(z_test, dim=0).to(device)
    labels2 = torch.argmax(z_test, dim=1).to(device)
    g_mod1 = g_mod1.to(device)
    g_mod2 = g_mod2.to(device)

    model.fit(g_mod1, g_mod2, labels1, labels2, train_size=train_size)
    model.load(f'models/model_{rndseed}.pth')

    test_idx = np.arange(train_size, g_mod1.num_nodes('cell'))
    print(model.predict(test_idx, enhance=True, batch1=batch_mod1, batch2=batch_mod2))
    print(model.score(test_idx, labels_matrix=z_test, enhance=True, batch1=batch_mod1, batch2=batch_mod2))
""" To reproduce scMoGCN on other samples, please refer to command lines belows:
GEX-ADT:
python scmogcn.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

GEX-ATAC:
python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

"""
