import argparse
import os

import anndata
import numpy as np
import torch

from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.scrgcl import scRGCLWrapper
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset", nargs="+", type=int, default=[138], help="Testing dataset IDs")
    parser.add_argument("--tissue", default="Brain", type=str)
    parser.add_argument("--train_dataset", nargs="+", type=int, default=[328], help="List of training dataset ids.")
    parser.add_argument("--val_size", type=float, default=0.0, help="val size")
    parser.add_argument("--species", default="human", type=str)
    parser.add_argument("--cache", action="store_true", help="Cache processed data")

    # scRGCL Specific parameters
    parser.add_argument('--quantile', type=float, default=0.99, help='Quantile threshold for network filtering')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory for models')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of repetitions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model parameters
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')

    args = parser.parse_args()
    runs = args.num_runs

    # Check GPU availability
    device = torch.device("cuda:" + str(args.gpu))
    print(f"Running on {device}")

    scores = []
    for run in range(runs):
        set_seed(args.seed + run)
        model = scRGCLWrapper(
            out_dir=args.out_dir,
            dropout_ratio=args.dropout_ratio,
            init_lr=args.init_lr,
            seed=args.seed + run,
            device=device,
        )

        preprocessing_pipeline = model.preprocessing_pipeline(thres=args.quantile, species=args.species)
        # 1. Load Data using DANCE
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)

        # 2. Extract Train/Test Data
        # return_type="torch" ensures we get tensors, but we might need numpy for AnnData creation
        x_train, y_train = data.get_train_data(return_type="torch")
        x_test, y_test = data.get_test_data(return_type="torch")

        # 3. Convert Labels (One-hot -> Index)
        # scRGCLWrapper internally expects 1D array of indices or strings
        y_train_indices = y_train.argmax(1).cpu().numpy()

        # 4. Construct AnnData for Training
        # The scRGCL wrapper fit method expects an AnnData object to run gen_data internally
        train_adata = anndata.AnnData(X=x_train.cpu().numpy())
        train_adata.uns = data.data.uns
        train_adata.obs['cell_type'] = y_train_indices

        # If gen_data relies on gene names (feature names), we should attach them.
        # DANCE data object usually holds feature info.
        if hasattr(data.data, "var_names"):
            train_adata.var_names = data.data.var_names

        # 5. Initialize Model
        # Note: device handling is largely done inside fit(), but we pass params here

        # 6. Fit the Model
        # Pass the constructed adata and specific args
        print(f"--- Run {run+1}/{runs}: Fitting model ---")
        model.fit(adata=train_adata, batch_size=args.batch_size)

        # 7. Evaluate
        # BaseMethod.score() will call self.predict(x_test) -> score_func(y_test, y_pred)
        # We pass x_test (Tensor or Numpy) which the wrapper handles in _prepare_test_loader
        print(f"--- Run {run+1}/{runs}: Evaluating ---")
        score = model.score(x_test, y_test, score_func="acc")
        scores.append(score)
        print(f"Run {run+1} Score: {score:.4f}")

    print(f"\nscRGCL {args.species} {args.tissue} Test Set {args.test_dataset}:")
    print(f"All Scores: {scores}")
    print(f"Mean: {np.mean(scores):.5f} +/- {np.std(scores):.5f}")
