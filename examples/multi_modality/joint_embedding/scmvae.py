import argparse

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn import preprocessing

from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmvae import scMVAE
from dance.transforms.preprocess import calculate_log_library_size


def parameter_setting():
    parser = argparse.ArgumentParser(description="Single cell Multi-omics data analysis")

    parser.add_argument("--workdir", "-wk", type=str, default="./new_test", help="work path")
    parser.add_argument("--outdir", "-od", type=str, default="./new_test", help="Output path")

    parser.add_argument("--lr", type=float, default=1E-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
    parser.add_argument("--eps", type=float, default=0.01, help="eps")

    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=200, help="Random seed for repeat results")
    parser.add_argument("--latent", "-l", type=int, default=10, help="latent layer dim")
    parser.add_argument("--max_epoch", "-me", type=int, default=25, help="Max epoches")
    parser.add_argument("--max_iteration", "-mi", type=int, default=3000, help="Max iteration")
    parser.add_argument("--anneal_epoch", "-ae", type=int, default=200, help="Anneal epoch")
    parser.add_argument("--epoch_per_test", "-ept", type=int, default=5,
                        help="Epoch per test, must smaller than max iteration.")
    parser.add_argument("--max_ARI", "-ma", type=int, default=-200, help="initial ARI")
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2")
    parser.add_argument("-device", "--device", default="cuda")
    parser.add_argument("--final_rate", type=float, default=1e-4)
    parser.add_argument("--scale_factor", type=float, default=4)

    return parser


if __name__ == "__main__":
    parser = parameter_setting()
    args = parser.parse_args()
    assert args.max_iteration > args.epoch_per_test

    dataset = JointEmbeddingNIPSDataset(args.subtask, root="./data/joint_embedding", preprocess="feature_selection")
    data = dataset.load_data()

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(data.mod["test_sol"].obs["cell_type"])
    data.mod["mod1"].obsm["labels"] = labels
    data.set_config(feature_mod=["mod1", "mod2"], label_mod="mod1", feature_channel_type=["layers", "layers"],
                    feature_channel=["counts", "counts"], label_channel="labels")

    (x_train, y_train), _ = data.get_train_data(return_type="torch")
    (x_test, y_test), labels = data.get_test_data(return_type="torch")

    lib_mean1, lib_var1 = calculate_log_library_size(np.concatenate([x_train.numpy(), x_test.numpy()]))
    lib_mean2, lib_var2 = calculate_log_library_size(np.concatenate([y_train.numpy(), y_test.numpy()]))
    lib_mean1 = torch.from_numpy(lib_mean1)
    lib_var1 = torch.from_numpy(lib_var1)
    lib_mean2 = torch.from_numpy(lib_mean2)
    lib_var2 = torch.from_numpy(lib_var2)

    Nfeature1 = x_train.shape[1]
    Nfeature2 = y_train.shape[1]

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
    train_size = len(data.get_split_idx("train"))
    train = data_utils.TensorDataset(x_train, lib_mean1[:train_size], lib_var1[:train_size], lib_mean2[:train_size],
                                     lib_var2[:train_size], y_train)

    valid = data_utils.TensorDataset(x_test, lib_mean1[train_size:], lib_var1[train_size:], lib_mean2[train_size:],
                                     lib_var2[train_size:], y_test)

    total = data_utils.TensorDataset(torch.cat([x_train, x_test]), torch.cat([y_train, y_test]))

    total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)
    model.init_gmm_params(total_loader)
    model.fit(args, train, valid, args.final_rate, args.scale_factor, device)

    embeds = model.predict(torch.cat([x_train, x_test]), torch.cat([y_train, y_test])).cpu().numpy()
    print(embeds)

    nmi_score, ari_score = model.score(x_test, y_test, labels)
    print(f"NMI: {nmi_score:.3f}, ARI: {ari_score:.3f}")
"""To reproduce scMVAE on other samples, please refer to command lines belows:

GEX-ADT:
python scmvae.py --subtask openproblems_bmmc_cite_phase2 --device cuda

GEX-ATAC:
python scmvae.py --subtask openproblems_bmmc_multiome_phase2 --device cuda

"""
