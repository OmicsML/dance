import argparse
import gc
import os
import pickle
import pprint
import sys
from math import e
from pathlib import Path
from typing import get_args

import numpy as np
import torch
import wandb
from sympy import elliptic_k

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scdeepsort import ScDeepSort
from dance.pipeline import PipelinePlaner, get_step3_yaml, run_step3, save_summary_data
from dance.transforms import FeatureCellPlaceHolder, FilterGenesTopK, WeightedFeatureSVD
from dance.transforms.cell_feature import WeightedFeaturePCA
from dance.transforms.filter import (
    FilterGenesNumberPlaceHolder,
    FilterGenesPlaceHolder,
    FilterGenesScanpyOrder,
    HighlyVariableGenesLogarithmizedByMeanAndDisp,
    HighlyVariableGenesLogarithmizedByTopGenes,
    HighlyVariableGenesRawCount,
)
from dance.transforms.graph.cell_feature_graph import CellFeatureGraph
from dance.transforms.misc import Compose, SetConfig
from dance.transforms.normalize import (
    ColumnSumNormalize,
    Log1P,
    NormalizePlaceHolder,
    NormalizeTotal,
    NormalizeTotalLog1P,
    ScTransform,
)
from dance.typing import LogLevel
from dance.utils import set_seed


def build_transform_pipeline(pipeline_name, num_genes=2208):
    transforms = []
    if pipeline_name == "atlas_scdeepsort_pancreas_53d208b0_scale_rawcount_svd":
        transforms.append(
            FilterGenesScanpyOrder(
                order=["min_counts", "max_counts", "max_cells", "min_cells"],
                min_counts=191,
                min_cells=0.03798547080750556,
                max_counts=0.9377398883656286,
                max_cells=0.964294533892298,
            ))
        transforms.append(ColumnSumNormalize(mode="l2", eps=0.7))
        transforms.append(HighlyVariableGenesRawCount(n_top_genes=5718, span=0.39461811670516034))
        transforms.append(
            WeightedFeatureSVD(
                n_components=591,
                out="feature.cell",
                log_level="INFO",
                feat_norm_mode="l2",
            ))
        transforms.append(CellFeatureGraph(cell_feature_channel="feature.cell"))
        transforms.append(SetConfig({"label_channel": "cell_type"}))
        return transforms

    pipeline_parts = pipeline_name.split("_")
    for i, pipline_part in enumerate(pipeline_parts):
        if pipline_part in {"FGPH", "FPGH"}:
            transforms.append(FilterGenesPlaceHolder())
        elif pipline_part == "NP":
            transforms.append(NormalizePlaceHolder())
        elif pipline_part == "Log":
            transforms.append(Log1P())
        elif pipline_part == "ST":
            transforms.append(ScTransform(processes_num=8))
        elif pipline_part == "NT":
            transforms.append(NormalizeTotal())
        elif pipline_part == "FK":
            transforms.append(FilterGenesTopK(num_genes=num_genes))
        elif pipline_part == "SF":
            transforms.append(ColumnSumNormalize())
        elif pipline_part == "FGNP":
            transforms.append(FilterGenesNumberPlaceHolder())
        elif pipline_part == "HvgRC":
            transforms.append(HighlyVariableGenesRawCount(n_top_genes=num_genes))
        elif pipline_part == "HvgLogTop":
            transforms.append(HighlyVariableGenesLogarithmizedByTopGenes(n_top_genes=num_genes))
        elif pipline_part == "NTLP":
            transforms.append(NormalizeTotalLog1P())
        elif pipline_part == "HVGmD":
            transforms.append(HighlyVariableGenesLogarithmizedByMeanAndDisp())
        elif pipline_part == "WPCA":
            transforms.append(
                WeightedFeaturePCA(out="feature.cell", log_level="INFO", save_info=True, split_name="train"))
        elif pipline_part == "WSVD":
            transforms.append(
                WeightedFeatureSVD(out="feature.cell", log_level="INFO", save_info=True, split_name="train"))
        elif pipline_part == "FP":
            transforms.append(FeatureCellPlaceHolder(out="feature.cell"))
    transforms.append(CellFeatureGraph(cell_feature_channel="feature.cell"))

    transforms.append(SetConfig({"label_channel": "cell_type"}))
    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    # parser.add_argument("--dense_dim", type=int, default=400, help="number of hidden gcn units")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--hidden_dim", type=int, default=200, help="number of hidden gcn units")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--n_layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--species", default="mouse", type=str)
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[], help="Testing dataset IDs")
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--tissue", default="Spleen", type=str)
    parser.add_argument("--train_dataset", nargs="+", default=[1970], help="List of training dataset ids.")
    parser.add_argument("--valid_dataset", nargs="+", default=None, help="List of valid dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.2, help="Validation split size in data loader.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--exp_num", type=int, default=1000)
    parser.add_argument("--root_path", default="/home/common/zyxing/dance/result/cta_scdeepsort", type=str)
    parser.add_argument("--data_dir", default="../temp_data", type=str)
    parser.add_argument("--filetype", default="csv")
    parser.add_argument("--pipeline_name", type=str, default="")
    parser.add_argument("--num_genes", type=int, default=2208)
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "2000"
    logger.info(f"Running ScDeepSort with the following parameters:\n{pprint.pformat(vars(args))}")
    file_root_path = Path(
        args.root_path, "_".join([
            "-".join([str(num) for num in dataset])
            for dataset in [args.train_dataset, args.valid_dataset, args.test_dataset]
            if (dataset is not None and dataset != [])
        ])).resolve()
    logger.info(f"\n files is saved in {file_root_path}")

    set_seed(args.seed)

    # Load data and perform necessary preprocessing
    data = CellTypeAnnotationDataset(species=args.species, tissue=args.tissue, test_dataset=args.test_dataset,
                                     train_dataset=args.train_dataset, valid_dataset=args.valid_dataset,
                                     data_dir=args.data_dir, val_size=args.val_size,
                                     filetype=args.filetype).load_data()
    if args.pipeline_name == "origin":
        preprocessing_pipeline = ScDeepSort.preprocessing_pipeline(normalize=True)
    else:
        # Prepare preprocessing pipeline and apply it to data
        transforms = build_transform_pipeline(args.pipeline_name, num_genes=args.num_genes)
        preprocessing_pipeline = Compose(*transforms, log_level="INFO")
    history = preprocessing_pipeline.transform_with_history(data)

    # Obtain training and testing data
    y_train = data.get_y(split_name="train", return_type="torch")
    y_valid = data.get_y(split_name="val", return_type="torch") if data.val_idx is not None else None
    y_test = data.get_y(split_name="test", return_type="torch")

    # Get cell feature graph for scDeepSort
    # TODO: make api for the following block?
    g = data.data.uns["CellFeatureGraph"]
    num_genes = data.shape[1]
    # Initialize model and get model specific preprocessing pipeline
    dense_dim = g.ndata["features"].shape[1]
    model = ScDeepSort(dense_dim, args.hidden_dim, args.n_layers, args.species, args.tissue, dropout=args.dropout,
                       batch_size=args.batch_size, device=args.device)

    gene_ids = torch.arange(num_genes)
    train_cell_ids = torch.LongTensor(data.train_idx) + num_genes
    test_cell_ids = torch.LongTensor(data.test_idx) + num_genes
    g_train = g.subgraph(torch.concat((gene_ids, train_cell_ids)))
    if data.val_idx is not None:
        valid_cell_ids = torch.LongTensor(data.val_idx) + num_genes
        g_valid = g.subgraph(torch.concat((gene_ids, valid_cell_ids)))
    else:
        g_valid = None
    g_test = g.subgraph(torch.concat((gene_ids, test_cell_ids)))

    # Train and evaluate models for several rounds
    def print_distribution(y_input, class_names=None):
        """通用分布打印函数 支持:

        1. One-hot格式 (Shape: [N, C]) -> 使用 sum(dim=0)
        2. Index格式   (Shape: [N])    -> 使用 bincount

        """
        # 0. 预处理：确保在 CPU 上并转为 long 类型 (bincount 需要 long)
        if y_input.is_cuda:
            y_input = y_input.cpu()

        # 1. 核心逻辑：自动判断输入类型并统计
        if y_input.ndim == 2:
            # One-hot 矩阵: 直接按列求和
            counts = y_input.sum(dim=0).long()
        else:
            # Index 向量: 使用 bincount 统计每个数字出现的次数
            # minlength 确保即使某些类别没出现，长度也能和 class_names 对齐
            num_classes = len(class_names) if class_names else (y_input.max() + 1).item()
            counts = torch.bincount(y_input.long(), minlength=num_classes)

        # 计算总数
        total = counts.sum().item()

        # 2. 打印表头
        print(f"{'ID':<5} {'Name':<20} {'Count':<8} {'Percentage'}")
        print("-" * 50)

        # 3. 循环输出
        for i, count in enumerate(counts):
            # 如果提供了名字列表，使用名字；否则用默认 ID
            # 防止名字列表比实际类别少导致越界
            if class_names and i < len(class_names):
                name = class_names[i]
            else:
                name = f"Class {i}"

            pct = count.item() / total if total > 0 else 0

            # 只打印存在或者列表里包含的类别
            # (如果你想隐藏数量为0的类别，可以在这里加 if count > 0: )
            print(f"{i:<5} {name:<20} {count.item():<8} {pct:.2%}")

    # Train and evaluate the model
    model.fit(g_train, y_train.argmax(1), epochs=args.n_epochs, lr=args.lr, weight_decay=args.weight_decay,
              val_ratio=args.test_rate)
    train_score = model.score(g_train, y_train)
    score = model.score(g_valid, y_valid) if g_valid is not None else None
    test_score = model.score(g_test, y_test)
    # with open(f"{file_root_path}/data{'_'+args.pipeline_name}.pkl", "wb") as f:  #base value is equal to expected value
    #     pickle.dump(history, f)
    pred_prob = model.predict_proba(g_test)
    pred_test = pred_prob.argmax(1)
    true_test = y_test.argmax(1).detach().cpu().numpy()
    label_names = np.asarray(list(data.data.obsm["cell_type"].columns), dtype=object)
    test_obs_names = np.asarray(data.data.obs_names[data.test_idx], dtype=object)
    output_prefix = f"{args.pipeline_name}_{args.tissue.lower()}".strip("_")
    np.savez_compressed(
        Path(__file__).resolve().parent / f"{output_prefix}_test_predictions.npz",
        predictions=pred_test,
        probabilities=pred_prob,
        true_labels=true_test,
        pred_label_names=label_names[pred_test],
        true_label_names=label_names[true_test],
        label_names=label_names,
        obs_names=test_obs_names,
    )
    print_distribution(torch.tensor(pred_test))
    print(f"Train accuracy: {train_score:.4f}")
    if score is not None:
        print(f"Validation accuracy: {score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
"""To reproduce the benchmarking results, please run the following command:

Mouse Brain
$ python scdeepsort.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python scdeepsort.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python scdeepsort.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:0 --pipeline_name FGPH_NP_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:1 --pipeline_name FGPH_NP_HvgRC_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:0 --pipeline_name FGPH_NP_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:2 --pipeline_name FGPH_Log_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:3 --pipeline_name FGPH_ST_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:4 --pipeline_name FGPH_NT_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:5 --pipeline_name FGPH_SF_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:5 --pipeline_name FGPH_NTLP_FGNP_WPCA
python main.py --species human --tissue Brain --train_dataset 328 --test_dataset 138 --device cuda:5 --pipeline_name FGPH_NTLP_HvgLogTop_WPCA

python main.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972  --test_dataset 245 332 377 398 405 455 470 492  --pipeline_name Log_WPCA --device cuda:0
python main.py --species human --tissue CD8 --train_dataset 1027 1357 1641 517 706 777 850 972  --test_dataset 245 332 377 398 405 455 470 492  --pipeline_name FPGH_Log_HvgRC_WSVD --device cuda:0


python main.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --pipeline_name WPCA --device cuda:1
python main.py --species human --tissue CD4 --train_dataset 1013 1247 598 732 767 768 770 784 845 864 --test_dataset 315 340 376 381 390 404 437 490 551 559 --pipeline_name WSVD --device cuda:1
"""
"""效果比较好的这些结果流程基本都会有所欠缺，其实人为构造一下，原理也是一样的."""
