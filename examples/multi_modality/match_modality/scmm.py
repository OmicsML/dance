import argparse

from dance.datasets.multimodality import ModalityMatchingDataset
from dance.modules.multi_modality.match_modality.scmm import MMVAE
from dance.utils.misc import default_parser_processor


@default_parser_processor(name="scmm", setup_torch_threads=True, default_torch_threads=1)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./modality_matching/output", help="outputs path")
    parser.add_argument("-d", "--data_folder", default="./data/modality_matching")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-t", "--subtask", default="openproblems_bmmc_cite_phase2_rna")
    parser.add_argument("--experiment", type=str, default="test", metavar="E", help="experiment name")
    parser.add_argument("--obj", type=str, default="m_elbo_naive_warmup", metavar="O",
                        help="objective to use (default: elbo)")
    parser.add_argument(
        "--llik_scaling", type=float, default=1., help="likelihood scaling for cub images/svhn modality when running in"
        "multimodal setting, set as 0 to use default value")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="batch size for data (default: 256)")
    parser.add_argument("--epochs", type=int, default=100, metavar="E", help="number of epochs to train (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="L", help="learning rate (default: 1e-3)")
    parser.add_argument("--latent_dim", type=int, default=10, metavar="L", help="latent dimensionality (default: 20)")
    parser.add_argument("--num_hidden_layers", type=int, default=2, metavar="H",
                        help="number of hidden layers in enc and dec (default: 2)")
    parser.add_argument("--r_hidden_dim", type=int, default=100, help="number of hidden units in enc/dec for gene")
    parser.add_argument("--p_hidden_dim", type=int, default=20,
                        help="number of hidden units in enc/dec for protein/peak")
    parser.add_argument("--pre_trained", type=str, default="",
                        help="path to pre-trained model (train from scratch if empty)")
    parser.add_argument("--learn_prior", type=bool, default=True, help="learn model prior parameters")
    parser.add_argument("--print_freq", type=int, default=0, metavar="f",
                        help="frequency with which to print stats (default: 0)")
    parser.add_argument("--deterministic_warmup", type=int, default=50, metavar="W", help="deterministic warmup")
    return parser


if __name__ == "__main__":
    args = parse_args()

    # Preprocess and load data
    dataset = ModalityMatchingDataset(args.subtask, root=args.data_folder, preprocess="feature_selection")
    data = dataset.load_data()

    # Set data config
    data.set_config(feature_mod=["mod1", "mod2"], label_mod="mod1", feature_channel_type=["layers", "layers"],
                    feature_channel=["counts", "counts"], label_channel="labels")

    (x_train, y_train), _ = data.get_train_data(return_type="torch")
    (x_test, y_test), labels = data.get_test_data(return_type="torch")

    args.r_dim = x_train.shape[1]
    args.p_dim = y_train.shape[1]

    model_class = "rna-protein" if args.subtask == "openproblems_bmmc_cite_phase2_rna" else "rna-dna"
    model = MMVAE(model_class, args).to(args.device)

    model.fit(x_train, y_train)
    print(model.predict(x_test, y_test))
    print(model.score(x_test, y_test, labels))
"""To reproduce scMM on other samples, please refer to command lines belows:

GEX-ADT (subset):
python scmm.py --subtask openproblems_bmmc_cite_phase2_rna_subset --device cuda

GEX-ADT:
python scmm.py --subtask openproblems_bmmc_cite_phase2_rna --device cuda

GEX-ATAC:
python scmm.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

"""
