import argparse

import numpy as np

from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdsc import ScDSC
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
    model_para = [512, 256, 256]
    # Cluster_para = [n_z1, n_z2, n_z3, n_init, n_input, n_clusters]
    Cluster_para = [256, 128, 32, 20, 100, 10]
    # Balance_para = [binary_crossentropy_loss, ce_loss, re_loss, zinb_loss, sigma]
    Balance_para = [1, 0.01, 0.1, 0.1, 1]

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--dataset", type=str, default="worm_neuron_cell",
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    # TODO: implement callbacks for "heat_kernel" and "cosine_normalized"
    parser.add_argument("--method", type=str, default="correlation", choices=["cosine", "correlation"])
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--n_enc_1", default=model_para[0], type=int)
    parser.add_argument("--n_enc_2", default=model_para[1], type=int)
    parser.add_argument("--n_enc_3", default=model_para[2], type=int)
    parser.add_argument("--n_dec_1", default=model_para[2], type=int)
    parser.add_argument("--n_dec_2", default=model_para[1], type=int)
    parser.add_argument("--n_dec_3", default=model_para[0], type=int)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n_z1", default=Cluster_para[0], type=int)
    parser.add_argument("--n_z2", default=Cluster_para[1], type=int)
    parser.add_argument("--n_z3", default=Cluster_para[2], type=int)
    parser.add_argument("--n_input", type=int, default=Cluster_para[4])
    parser.add_argument("--n_clusters", type=int, default=Cluster_para[5])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--v", type=int, default=1)
    parser.add_argument("--nb_genes", type=int, default=2000)
    parser.add_argument("--binary_crossentropy_loss", type=float, default=Balance_para[0])
    parser.add_argument("--ce_loss", type=float, default=Balance_para[1])
    parser.add_argument("--re_loss", type=float, default=Balance_para[2])
    parser.add_argument("--zinb_loss", type=float, default=Balance_para[3])
    parser.add_argument("--sigma", type=float, default=Balance_para[4])
    parser.add_argument("--seed", type=int, default=0, help="Initial seed random, offset for each repeatition")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    args = parser.parse_args()
    aris = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # Load data and perform necessary preprocessing
        dataloader = ClusteringDataset(args.data_dir, args.dataset)
        preprocessing_pipeline = ScDSC.preprocessing_pipeline(n_top_genes=args.nb_genes, n_neighbors=args.topk)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # inputs: adj, x, x_raw, n_counts
        inputs, y = data.get_data(return_type="default")
        args.n_input = inputs[1].shape[1]
        n_clusters = len(np.unique(y))

        model = ScDSC(
            pretrain_path=f"scdsc_{args.dataset}_pre.pkl",
            sigma=args.sigma,
            n_enc_1=args.n_enc_1,
            n_enc_2=args.n_enc_2,
            n_enc_3=args.n_enc_3,
            n_dec_1=args.n_dec_1,
            n_dec_2=args.n_dec_2,
            n_dec_3=args.n_dec_3,
            n_z1=args.n_z1,
            n_z2=args.n_z2,
            n_z3=args.n_z3,
            n_clusters=n_clusters,  #args.n_clusters,
            n_input=args.n_input,
            v=args.v,
            device=args.device)

        # Build and train model
        model.fit(inputs, y, lr=args.lr, epochs=args.epochs, bcl=args.binary_crossentropy_loss, cl=args.ce_loss,
                  rl=args.re_loss, zl=args.zinb_loss, pt_epochs=args.pretrain_epochs, pt_batch_size=args.batch_size,
                  pt_lr=args.pretrain_lr)

        # Evaluate model predictions
        score = model.score(None, y)
        print(f"{score=:.4f}")
        aris.append(score)

    print('scdsc')
    print(args.dataset)
    print(f'aris: {aris}')
    print(f'aris: {np.mean(aris)} +/- {np.std(aris)}')
"""Reproduction information
10X PBMC:
python scdsc.py --dataset 10X_PBMC --sigma 0.5 --topk 10 --pretrain_epochs 100 --v 3 --n_enc_1 1024 --n_enc_3 64 --n_dec_1 64 --n_z1 64

Mouse Bladder:
python scdsc.py --dataset mouse_bladder_cell --sigma 0.5 --topk 50 --pretrain_epochs 100 --v 7

Mouse ES:
python scdsc.py --dataset mouse_ES_cell --sigma 0.1 --topk 10 --pretrain_epochs 50 --v 2

Worm Neuron:
python scdsc.py --dataset worm_neuron_cell --sigma 0.5 --topk 10 --pretrain_epochs 100 --v 3 --n_enc_3 64 --n_dec_1 64 --n_z1 64 --n_z2 64
"""
