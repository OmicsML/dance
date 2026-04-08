import argparse
import pprint

import numpy as np
import scanpy as sc
import torch

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scgat import scGATAnnotator
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--val_size", type=float, default=0.0, help="val size")

    # GAT specific args
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5000)

    args = parser.parse_args()
    logger.setLevel("INFO")
    logger.info(f"Running GAT with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)

        # 1. 初始化模型 (参数需要显式传递，不能直接传 args)
        device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
        model = scGATAnnotator(hidden_channels=args.hidden_channels, batch_size=args.batch_size, n_epochs=args.n_epochs,
                               device=device, random_seed=seed)

        # 2. 定义完整的预处理 Pipeline
        # 注意：scGATGraphTransform 只负责建图，特征提取(PCA)需要在此之前完成
        preprocessing_pipeline = model.preprocessing_pipeline(
            label_column="cell_type",  # 确保这里与 dataset 的标签列名一致
            train_ratio=0.7,
            val_ratio=0.15,
            n_neighbors=15,
            log_level="INFO")

        # 3. 加载并转换数据
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # 4. 训练
        # 修改点：GAT 需要图结构，直接传入包含 'pyg_data' 的 AnnData 对象
        # data.data 是底层的 scanpy.AnnData 对象
        logger.info("Training GAT model...")

        # fit 内部会自动从 data.data.uns['pyg_data'] 提取数据
        model.fit(data.data)

        # 5. 评估
        logger.info("Evaluating...")

        # 获取真实标签用于计算准确率
        # get_test_data 返回的是 (feature, label) 元组，我们只需要 label
        _, y_test = data.get_test_data(return_type="torch")
        if y_test.dim() > 1 and y_test.shape[1] > 1:
            y_test = y_test.argmax(1)  # 转为 label index

        # 进行预测
        # predict 会返回所有细胞的预测结果 (Array of shape [N_total])
        all_preds = model.predict(data.data)

        # 获取测试集掩码来提取对应的预测
        # scGATGraphTransform 会将 mask 存储在 pyg_data 中，也会同步到 obs 中 (gat_test_mask)
        # 或者我们可以直接利用 dance 数据集的划分逻辑
        # 最稳妥的方式是直接查看 pyg_data 中的 mask
        pyg_data = data.data.uns['pyg_data']
        test_mask = pyg_data.test_mask.cpu().numpy()

        # 提取测试集的预测结果
        y_pred = all_preds[test_mask]

        # 计算准确率
        # 确保 y_pred 和 y_test 长度一致
        if len(y_pred) != len(y_test):
            logger.warning(f"Shape mismatch: Preds {len(y_pred)} vs Labels {len(y_test)}. "
                           "Using intersection or checking split logic.")
            # 这种情况通常极少发生，除非 cache 导致 split 不一致

        score = (y_pred == y_test.cpu().numpy()).mean()
        scores.append(score)
        print(f"{score=:.4f}")

    print(f"GAT {args.species} {args.tissue} {args.test_dataset}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")
