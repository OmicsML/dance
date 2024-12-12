import argparse

import numpy as np
import pandas as pd
import torch

from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.jae import JAEWrapper
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="GSE140203_SKIN_atac2gex",
                        choices=["openproblems_bmmc_cite_phase2", "openproblems_bmmc_multiome_phase2","GSE140203_BRAIN_atac2gex","GSE140203_SKIN_atac2gex"])
    parser.add_argument("-d", "--data_folder", default="./data/joint_embedding")
    parser.add_argument("-pre", "--pretrained_folder", default="./data/joint_embedding/pretrained")
    parser.add_argument("-csv", "--csv_path", default="decoupled_lsi.csv")
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-nm", "--normalize", default=1, type=int, choices=[0, 1])
    parser.add_argument("--runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--span", type=float, default=0.3)

    args = parser.parse_args()

    device = args.device
    pre_normalize = bool(args.normalize)
    torch.set_num_threads(args.cpus)
    rndseed = args.seed
    set_seed(rndseed)

    dataset = JointEmbeddingNIPSDataset(args.subtask, root=args.data_folder, preprocess="pca", normalize=True,span=args.span)
    data = dataset.load_data()

    data.set_config(
        feature_mod=["mod1", "mod2"],
        label_mod=["mod1", "mod1", "mod1", "mod1", "mod1"],
        feature_channel=["X_pca", "X_pca"],
        label_channel=["cell_type", "batch_label", "phase_labels", "S_scores", "G2M_scores"],
    )
    if True:
        cell_type_labels = data.data['test_sol'].obs["cell_type"].to_numpy()
        cell_type_labels_unique = list(np.unique(cell_type_labels))
        c_labels = np.array([cell_type_labels_unique.index(item) for item in cell_type_labels])
        data.data['mod1'].obsm["cell_type"] = c_labels
        data.data["mod1"].obsm["S_scores"] = np.zeros(data.data['mod1'].shape[0])
        data.data["mod1"].obsm["G2M_scores"] = np.zeros(data.data['mod1'].shape[0])
        data.data["mod1"].obsm["batch_label"] = np.zeros(data.data['mod1'].shape[0])
        data.data["mod1"].obsm["phase_labels"] = np.zeros(data.data['mod1'].shape[0])
    (X_mod1_train, X_mod2_train), (cell_type, batch_label, phase_label, S_score,
                                   G2M_score) = data.get_train_data(return_type="torch")
    (X_mod1_test, X_mod2_test), (cell_type_test, _, _, _, _) = data.get_test_data(return_type="torch")
    X_train = torch.cat([X_mod1_train, X_mod2_train], dim=1)
    phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)
    X_test = torch.cat([X_mod1_test, X_mod2_test], dim=1)
    X_test = torch.cat([X_train, X_test]).float().to(device)
    test_id = np.arange(X_test.shape[0])
    labels = torch.cat([cell_type, cell_type_test]).numpy()
    adata_sol = data.data['test_sol']  # [data._split_idx_dict['test']]

    res = None
    for k in range(args.runs):
        set_seed(args.seed + k)
        model = JAEWrapper(args, num_celL_types=int(cell_type.max() + 1), num_batches=int(batch_label.max() + 1),
                           num_phases=phase_score.shape[1], num_features=X_train.shape[1])
        model.fit(X_train, cell_type, batch_label, phase_score, max_epochs=50)

        embeds = model.predict(X_test, test_id).cpu().numpy()
        print(embeds)

        score = model.score(X_test, test_id, labels, metric="clustering")
        # score.update(model.score(X_test, test_id, labels, adata_sol=adata_sol, metric="openproblems"))
        score.update({
            'seed': args.seed + k,
            'subtask': args.subtask,
            'method': 'jae',
        })

        if res is not None:
            res = res.append(score, ignore_index=True)
        else:
            for s in score:
                score[s] = [score[s]]
            res = pd.DataFrame(score)
    print(res)
"""To reproduce JAE on other samples, please refer to command lines belows:

GEX-ADT:
$ python jae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
$ python jae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
