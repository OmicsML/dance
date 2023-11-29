import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dance.datasets.multimodality import ModalityMatchingDataset
from dance.modules.multi_modality.match_modality.scmogcn import ScMoGCNWrapper
from dance.transforms.graph.cell_feature_graph import CellFeatureBipartiteGraph
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("-d", "--data_folder", default="./data/modality_matching")
    parser.add_argument("-csv", "--csv_path", default="decoupled_lsi.csv")
    parser.add_argument("-l", "--layers", default=4, type=int, choices=[3, 4, 5, 6, 7])
    parser.add_argument("-lr", "--learning_rate", default=6e-4, type=float)
    parser.add_argument("-dis", "--disable_propagation", default=0, type=int, choices=[0, 1, 2])
    parser.add_argument("-aux", "--auxiliary_loss", default=True, type=bool)
    parser.add_argument("-pk", "--pickle_suffix", default="_lsi_input_pca_count.pkl")
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-e", "--epochs", default=2000, type=int)
    parser.add_argument("-tq", "--threshold_quantile", default=0.95, type=float)
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")

    args = parser.parse_args()

    torch.set_num_threads(args.cpus)
    rndseed = args.seed
    set_seed(rndseed)

    subtask = args.subtask
    data_folder = args.data_folder
    device = args.device
    layers = args.layers
    pkl_path = subtask + args.pickle_suffix

    # Preprocess and load data
    dataset = ModalityMatchingDataset(subtask, root=data_folder, preprocess="pca", pkl_path=pkl_path)
    data = dataset.load_data()

    # ScMoGNN graph construction
    data = CellFeatureBipartiteGraph(cell_feature_channel="X_pca", mod="mod1")(data)
    data = CellFeatureBipartiteGraph(cell_feature_channel="X_pca", mod="mod2")(data)

    data.set_config(feature_mod=["mod1", "mod2", "mod1", "mod2"], feature_channel_type=["uns", "uns", "obs", "obs"],
                    feature_channel=["g", "g", "batch", "batch"], label_mod="mod1", label_channel="labels")
    (g_mod1, g_mod2, batch_mod1, batch_mod2), z = data.get_data(return_type="default")

    if subtask == "openproblems_bmmc_cite_phase2_rna":
        LATENT_SIZE = 64
        TEMPERATURE = 2.739896
        LAYERS = [
            [(g_mod1.num_nodes("feature"), 512, 0.25), (512, 512, 0.25), (512, LATENT_SIZE)],
            [(g_mod2.num_nodes("feature"), 512, 0.2), (512, 512, 0.2), (512, LATENT_SIZE)],
            [(LATENT_SIZE, 512, 0.2), (512, g_mod1.num_nodes("feature"))],
            [(LATENT_SIZE, 512, 0.2), (512, g_mod2.num_nodes("feature"))],
        ]
    else:
        LATENT_SIZE = 256
        TEMPERATURE = 3.065016
        LAYERS = [
            [(g_mod1.num_nodes("feature"), 1024, 0.5), (1024, 1024, 0.5), (1024, LATENT_SIZE)],
            [(g_mod2.num_nodes("feature"), 2048, 0.5), (2048, LATENT_SIZE)],
            [(LATENT_SIZE, 512, 0.2), (512, g_mod1.num_nodes("feature"))],
            [(LATENT_SIZE, 512, 0.2), (512, g_mod2.num_nodes("feature"))],
        ]

    train_size = len(data.get_split_idx("train"))
    z_test = F.one_hot(torch.from_numpy(z[train_size:]).long())
    labels1 = torch.argmax(z_test, dim=0).to(device)
    labels2 = torch.argmax(z_test, dim=1).to(device)
    g_mod1 = g_mod1.to(device)
    g_mod2 = g_mod2.to(device)
    test_idx = np.arange(train_size, g_mod1.num_nodes("cell"))

    res = pd.DataFrame({'score': [], 'seed': [], 'subtask': [], 'method': []})
    for k in range(args.runs):
        set_seed(args.seed + k)
        model = ScMoGCNWrapper(
            args,
            LAYERS,
            TEMPERATURE,
        )
        model.fit(g_mod1, g_mod2, labels1, labels2, train_size=train_size)
        #     model.load(f"models/model_{rndseed}.pth")

        pred = model.predict(test_idx, enhance=True, batch1=batch_mod1, batch2=batch_mod2,
                             threshold_quantile=args.threshold_quantile)
        score = model.score(test_idx, labels_matrix=z_test, enhance=True, batch1=batch_mod1, batch2=batch_mod2,
                            threshold_quantile=args.threshold_quantile)

        print(pred)
        res = res.append({
            'score': score,
            'seed': k,
            'subtask': args.subtask,
            'method': 'scmogcn',
        }, ignore_index=True)
    print(res)
"""To reproduce scMoGCN on other samples, please refer to command lines belows:

GEX-ADT (subset):
$ python scmogcn.py --subtask openproblems_bmmc_cite_phase2_rna_subset --threshold_quantile 0.85 --device cuda

GEX-ADT:
$ python scmogcn.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

GEX-ATAC:
$ python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

"""
