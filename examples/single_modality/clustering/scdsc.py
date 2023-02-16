import argparse
import os
from argparse import Namespace
from time import time

from dance.data import Data
from dance.datasets.singlemodality import ClusteringDataset
from dance.modules.single_modality.clustering.scdsc import SCDSCWrapper
from dance.utils import set_seed

# for repeatability
set_seed(42)

if __name__ == "__main__":
    time_start = time()
    parser = argparse.ArgumentParser()

    # model_para = [n_enc_1(n_dec_3), n_enc_2(n_dec_2), n_enc_3(n_dec_1)]
    model_para = [512, 256, 256]
    # Cluster_para = [n_z1, n_z2, n_z3, n_init, n_input, n_clusters]
    Cluster_para = [256, 128, 32, 20, 100, 10]
    # Balance_para = [binary_crossentropy_loss, ce_loss, re_loss, zinb_loss, sigma]
    Balance_para = [1, 0.01, 0.1, 0.1, 1]

    parser.add_argument("--name", type=str, default="worm_neuron_cell",
                        choices=["10X_PBMC", "mouse_bladder_cell", "mouse_ES_cell", "worm_neuron_cell"])
    parser.add_argument("--pretrain_path", type=str, default="worm_neuron_cell")
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
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_z1", default=Cluster_para[0], type=int)
    parser.add_argument("--n_z2", default=Cluster_para[1], type=int)
    parser.add_argument("--n_z3", default=Cluster_para[2], type=int)
    parser.add_argument("--n_input", type=int, default=Cluster_para[4])
    parser.add_argument("--n_clusters", type=int, default=Cluster_para[5])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--v", type=int, default=1)
    parser.add_argument("--nb_genes", type=int, default=2000)
    parser.add_argument("--binary_crossentropy_loss", type=float, default=Balance_para[0])
    parser.add_argument("--ce_loss", type=float, default=Balance_para[1])
    parser.add_argument("--re_loss", type=float, default=Balance_para[2])
    parser.add_argument("--zinb_loss", type=float, default=Balance_para[3])
    parser.add_argument("--sigma", type=float, default=Balance_para[4])
    args = parser.parse_args()

    # File = [gene_expresion data file, Graph file, h5 file, pretrain_path]
    File = [
        os.path.join("data", args.name),
        None,
        os.path.join("data", f"{args.name}.h5"),
        os.path.join("model", f"{args.name}_pre.pkl"),
    ]
    args.pretrain_path = File[3]
    if not os.path.exists("./graph/"):
        os.makedirs("./graph/")
    if not os.path.exists("./model/"):
        os.makedirs("./model/")

    adata, labels = ClusteringDataset("./data", args.name).load_data()
    adata.obsm["Group"] = labels
    data = Data(adata, train_size="all")

    preprocessing_pipeline = SCDSCWrapper.preprocessing_pipeline(n_top_genes=args.nb_genes, n_neighbors=args.topk)
    preprocessing_pipeline(data)

    (x, x_raw, n_counts, adj), y = data.get_data(return_type="default")
    args.n_input = x.shape[1]

    # Pretrain AE
    model = SCDSCWrapper(Namespace(**vars(args)))
    if not os.path.exists(args.pretrain_path):
        model.pretrain_ae(x, args.batch_size, args.pretrain_epochs, args.pretrain_path)

    # Train scDSC
    model.fit(x, y, x_raw, n_counts, adj, lr=args.lr, n_epochs=args.n_epochs, bcl=args.binary_crossentropy_loss,
              cl=args.ce_loss, rl=args.re_loss, zl=args.zinb_loss)
    print(f"Running Time：{int(time() - time_start)} seconds")

    y_pred = model.predict()
    print(f"Prediction (first ten): {y_pred[:10]}")
    acc, nmi, ari = model.score(y)
    print("ACC: {:.4f}, NMI: {:.4f}, ARI: {:.4f}".format(acc, nmi, ari))
"""Reproduction information
10X PBMC:
python scdsc.py --name 10X_PBMC --method cosine --topk 30 --v 7 --binary_crossentropy_loss 0.75 --ce_loss 0.5 --re_loss 0.1 --zinb_loss 2.5 --sigma 0.4

Mouse Bladder:
python scdsc.py --name mouse_bladder_cell --topk 50 --v 7 --binary_crossentropy_loss 2.5 --ce_loss 0.1 --re_loss 0.5 --zinb_loss 1.5 --sigma 0.6

Mouse ES:
python scdsc.py --name mouse_ES_cell --topk 50 --v 7 --binary_crossentropy_loss 0.1 --ce_loss 0.01 --re_loss 1.5 --zinb_loss 0.5 --sigma 0.1

Worm Neuron:
python scdsc.py --name worm_neuron_cell --topk 20 --v 7 --binary_crossentropy_loss 2 --ce_loss 2 --re_loss 3 --zinb_loss 0.1 --sigma 0.4
"""
