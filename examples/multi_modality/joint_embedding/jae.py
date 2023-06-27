import argparse
import random

import numpy as np
import torch

from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.jae import JAEWrapper
from dance.utils import set_seed

if __name__ == "__main__":
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2",
                        choices=["openproblems_bmmc_cite_phase2", "openproblems_bmmc_multiome_phase2"])
    parser.add_argument("-d", "--data_folder", default="./data/joint_embedding")
    parser.add_argument("-pre", "--pretrained_folder", default="./data/joint_embedding/pretrained")
    parser.add_argument("-csv", "--csv_path", default="decoupled_lsi.csv")
    parser.add_argument("-seed", "--rnd_seed", default=rndseed, type=int)
    parser.add_argument("-cpu", "--cpus", default=1, type=int)
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-nm", "--normalize", default=1, type=int, choices=[0, 1])

    args = parser.parse_args()

    device = args.device
    pre_normalize = bool(args.normalize)
    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)

    dataset = JointEmbeddingNIPSDataset(args.subtask, root=args.data_folder, preprocess="aux", normalize=True)
    data = dataset.load_data()

    data.set_config(
        feature_mod=["mod1", "mod2"],
        label_mod=["mod1", "mod1", "mod1", "mod1", "mod1"],
        feature_channel=["X_pca", "X_pca"],
        label_channel=["cell_type", "batch_label", "phase_labels", "S_scores", "G2M_scores"],
    )
    (X_mod1_train, X_mod2_train), (cell_type, batch_label, phase_label, S_score,
                                   G2M_score) = data.get_train_data(return_type="torch")
    (X_mod1_test, X_mod2_test), (cell_type_test, _, _, _, _) = data.get_test_data(return_type="torch")
    X_train = torch.cat([X_mod1_train, X_mod2_train], dim=1)
    phase_score = torch.cat([S_score[:, None], G2M_score[:, None]], 1)
    model = JAEWrapper(args, num_celL_types=int(cell_type.max() + 1), num_batches=int(batch_label.max() + 1),
                       num_phases=phase_score.shape[1], num_features=X_train.shape[1])
    model.fit(X_train, cell_type, batch_label, phase_score)
    model.load(f"models/model_joint_embedding_{rndseed}.pth")

    with torch.no_grad():
        X_test = torch.cat([X_mod1_test, X_mod2_test], dim=1).float().to(device)
        test_id = np.arange(X_test.shape[0])
        labels = cell_type_test.numpy()
        embeds = model.predict(X_test, test_id).cpu().numpy()
        print(embeds)
        print(model.score(X_test, test_id, labels, metric="clustering"))
"""To reproduce JAE on other samples, please refer to command lines belows:

GEX-ADT:
python jae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
python jae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
