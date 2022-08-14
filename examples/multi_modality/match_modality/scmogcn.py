import argparse
import random

import numpy as np
import torch

from dance.datasets.multimodality import ModalityMatchingDataset
from dance.modules.multi_modality.match_modality.scmogcn import ScMoGCNWrapper
from dance.transforms.graph_construct import basic_feature_propagation
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

    if subtask == 'openproblems_bmmc_cite_phase2_rna':
        HIDDEN_SIZE = 64
        TEMPERATURE = 2.739896
        model = ScMoGCNWrapper(args, [[
            (dataset.preprocessed_features['mod1_train'].shape[1], 512, 0.25), (512, 512, 0.25), (512, HIDDEN_SIZE)
        ], [(dataset.preprocessed_features['mod2_train'].shape[1], 512, 0.2), (512, 512, 0.2),
            (512, HIDDEN_SIZE)], [(HIDDEN_SIZE, 512, 0.2), (512, dataset.preprocessed_features['mod1_train'].shape[1])],
                                      [(HIDDEN_SIZE, 512, 0.2),
                                       (512, dataset.preprocessed_features['mod2_train'].shape[1])]], TEMPERATURE)
    else:
        HIDDEN_SIZE = 256
        TEMPERATURE = 3.065016
        model = ScMoGCNWrapper(args, [[
            (dataset.preprocessed_features['mod1_train'].shape[1], 1024, 0.5), (1024, 1024, 0.5), (1024, HIDDEN_SIZE)
        ], [(dataset.preprocessed_features['mod2_train'].shape[1], 2048, 0.5),
            (2048, HIDDEN_SIZE)], [(HIDDEN_SIZE, 512, 0.2),
                                   (512, dataset.preprocessed_features['mod1_train'].shape[1])],
                                      [(HIDDEN_SIZE, 512, 0.2),
                                       (512, dataset.preprocessed_features['mod2_train'].shape[1])]], TEMPERATURE)

    hcell_mod1, hcell_mod2 = basic_feature_propagation(dataset, layers, device=device)
    z_test = torch.from_numpy(dataset.test_sol.X.toarray())
    labels1 = torch.argmax(z_test, dim=1).to(device)
    labels0 = torch.argmax(z_test, dim=0).to(device)

    model.fit(dataset, [hcell_mod1, hcell_mod2], [labels0, labels1])
    model.load(f'models/model_{rndseed}.pth')

    test_inputs = [hcell_mod1, hcell_mod2]
    test_idx = np.arange(dataset.sparse_features()[0].shape[0],
                         dataset.sparse_features()[0].shape[0] + dataset.sparse_features()[2].shape[0])
    print(model.predict(test_inputs, test_idx, enhance=True, dataset=dataset))
    print(model.score(test_inputs, test_idx, z_test, enhance=True, dataset=dataset))
""" To reproduce scMoGCN on other samples, please refer to command lines belows:
GEX-ADT:
python scmogcn.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

GEX-ATAC:
python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

"""
