import argparse

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scheteronet import (
    convert_dgl_to_original_format,
    eval_acc,
    print_statistics,
    scHeteroNet,
    set_graph_split,
    set_split,
)
from dance.transforms.misc import Compose, SaveRaw
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[1759], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[1970], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="val size")
    parser.add_argument("--species", default="mouse", type=str)

    parser.add_argument('--data_dir', type=str, default='../temp_data')

    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_step', type=int, default=1, help='how often to print')
    parser.add_argument("--num_runs", type=int, default=1, help="Number of repetitions")
    parser.add_argument('--train_prop', type=float, default=.6, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.2, help='validation label proportion')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')
    parser.add_argument('--knn_num', type=int, default=5, help='number of k for KNN graph')
    parser.add_argument('--T', type=float, default=1.0, help='temperature for Softmax')

    # hyper-parameter for model arch and training
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers for deep methods')
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='number of mlp layers')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--m_in', type=float, default=-5, help='upper bound for in-distribution energy')
    parser.add_argument('--m_out', type=float, default=-1, help='lower bound for in-distribution energy')
    parser.add_argument('--use_prop', action='store_true', help='whether to use energy belief propagation')
    parser.add_argument('--oodprop', type=int, default=2, help='number of layers for energy belief propagation')
    parser.add_argument('--oodalpha', type=float, default=0.3, help='weight for residual connection in propagation')
    parser.add_argument('--use_zinb', action='store_true',
                        help='whether to use ZINB loss (use if you do not need this)')
    parser.add_argument('--use_2hop', action='store_false',
                        help='whether to use 2-hop propagation (use if you do not need this)')
    parser.add_argument('--zinb_weight', type=float, default=1e-4)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    # display and utility
    parser.add_argument('--display_step', type=int, default=10, help='how often to print')
    parser.add_argument('--print_prop', action='store_true', help='print proportions of predicted class')
    parser.add_argument('--print_args', action='store_true', help='print args for hyper-parameter searching')
    parser.add_argument('--cl_weight', type=float, default=0.0)
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument('--spatial', action='store_false', help='read spatial')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = args.num_runs
    results = [[] for _ in range(runs)]
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    eval_func = eval_acc
    for run in range(runs):
        set_seed(args.seed + run)
        dataloader = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                               train_dataset=args.train_dataset, data_dir=args.data_dir,
                                               val_size=args.val_size)
        ref_data_name = f"{args.species}_{args.tissue}_{args.train_dataset}"
        preprocessing_pipeline = scHeteroNet.preprocessing_pipeline()
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        set_split(data, data.train_idx, data.val_idx, data.test_idx)
        # dataset_ind, dataset_ood_tr, dataset_ood_te, adata = load_cell_graph_fixed(
        #     data.data, ref_data_name)
        g = data.data.uns['HeteronetGraph']
        # set_graph_split(data.data,ref_data_name,g)
        dataset_ind, dataset_ood_tr, dataset_ood_te, adata = convert_dgl_to_original_format(g, data.data, ref_data_name)
        if len(dataset_ind.y.shape) == 1:
            dataset_ind.y = dataset_ind.y.unsqueeze(1)
        if len(dataset_ood_tr.y.shape) == 1:
            dataset_ood_tr.y = dataset_ood_tr.y.unsqueeze(1)
        if isinstance(dataset_ood_te, list):
            for single_dataset_ood_te in dataset_ood_te:
                if len(single_dataset_ood_te.y.shape) == 1:
                    single_dataset_ood_te.y = single_dataset_ood_te.y.unsqueeze(1)
        else:
            if len(dataset_ood_te.y.shape) == 1:
                dataset_ood_te.y = dataset_ood_te.y.unsqueeze(1)

        c = max(dataset_ind.y.max().item() + 1, dataset_ind.y.shape[1])
        d = dataset_ind.graph['node_feat'].shape[1]
        model = scHeteroNet(d, c, dataset_ind.edge_index.to(device), dataset_ind.num_nodes,
                            hidden_channels=args.hidden_channels, num_layers=args.num_layers, dropout=args.dropout,
                            use_bn=args.use_bn, device=device, min_loss=100000)
        criterion = nn.NLLLoss()
        model.train()
        model.reset_parameters()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            loss = model.fit(dataset_ind, dataset_ood_tr, args.use_zinb, adata, args.zinb_weight, args.cl_weight,
                             args.mask_ratio, criterion, optimizer)
            # model.evaluate(dataset_ind, dataset_ood_te, criterion, eval_func, args.display_step, run, results, epoch,
            #                loss, f"{args.species}_{args.tissue}_{args.train_dataset}", args.T, args.use_prop,
            #                args.use_2hop, args.oodprop, args.oodalpha)
            # test_idx=dataset_ind.splits['test']
            test_score = model.score(dataset_ind, dataset_ind.y, data.test_idx)
            print(test_score)

#TODO test_score is true delete odd test以及其他评估方法，再次测试，然后将valid等和其他算法保持一致。
"""

Epoch: 00, Loss: 2.3264, AUROC: 44.03%, AUPR: 99.12%, FPR95: 100.00%, Test Score: 46.85%
Run 01:
Chosen epoch: 4
OOD Test 1 Final AUROC: 64.61
OOD Test 1 Final AUPR: 99.53
OOD Test 1 Final FPR95: 100.00
IND Test Score: 81.88
All runs:
OOD Test 1 Final AUROC: 64.61
OOD Test 1 Final AUPR: 99.53
OOD Test 1 Final FPR: 100.00
IND Test Score: 81.88

python scheteronet.py --gpu -1 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue Brain --train_dataset 328 --test_dataset 138 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972 --test_dataset 245 332 377 398 405 455 470 492 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue Spleen --train_dataset 3043 3777 4029 4115 4362 4657 --test_dataset 138 --use_zinb --use_prop --use_2hop

python scheteronet.py --gpu -1  --species human --tissue Immune --train_dataset 11407 1519 636 713 9054 9258 --test_dataset 138 --use_zinb --use_prop --use_2hop
"""
