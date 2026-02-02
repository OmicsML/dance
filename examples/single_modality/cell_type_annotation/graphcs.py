import argparse
import pprint
from typing import get_args

import numpy as np
import torch

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.graphcs import GraphCSClassifier, load_GBP_data
from dance.typing import LogLevel
from dance.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Base Dance arguments
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, set to -1 for CPU")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[1759], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Spleen")
    parser.add_argument("--train_dataset", nargs="+", default=[1970], type=int, help="list of dataset id")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--val_size", type=float, default=0.0, help="val size")
    
    # Training parameters (passed to GraphCSClassifier.__init__)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--vat_lr", type=float, default=0.1, help="VAT learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    
    # Model architecture parameters
    parser.add_argument("--layer", type=int, default=2, help="number of layers")
    parser.add_argument("--hidden", type=int, default=256, help="hidden dimensions")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--bias", default='none', help="bias usage")
    
    # GraphCS specific parameters (if used in preprocessing or internal logic)
    parser.add_argument("--alpha", type=float, default=0.05, help="decay factor (GraphCS)")
    parser.add_argument("--rmax", type=float, default=1e-5, help="threshold (GraphCS)")
    parser.add_argument("--rrz", type=float, default=0.5, help="gamma/rrz (GraphCS)")

    args = parser.parse_args()
    
    # Update GPU argument for the model wrapper expectations
    # The class expects args.gpus to be a list
    args.gpus = [args.gpu] if args.gpu != -1 else []
    
    logger.setLevel(args.log_level)
    logger.info(f"Running GraphCS with the following parameters:\n{pprint.pformat(vars(args))}")

    scores = []
    for seed in range(args.seed, args.seed + args.num_runs):
        set_seed(seed)
        
        # Initialize model: args are passed here, so self.batch_size etc are set now
        model = GraphCSClassifier(args, random_state=seed)
        
        # Get preprocessing pipeline
        preprocessing_pipeline = model.preprocessing_pipeline(log_level=args.log_level)
        
        dataloader = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                               species=args.species, tissue=args.tissue, val_size=args.val_size)
        
        data = dataloader.load_data(transform=preprocessing_pipeline, cache=args.cache)
        X,y=data.get_data(return_type="torch")
        temp_graph=data.data.uns["temp_graph"]
        dataset_name=f"{args.species}_{args.tissue}_{args.train_dataset}"
        features=load_GBP_data(dataset_name, args.alpha, args.rmax, args.rrz, X, temp_graph)
        # Obtain training and testing data
        x_train=features[data.train_idx]
        x_test=features[data.test_idx]
        y_train=y[data.train_idx]
        y_test=y[data.test_idx]
        
        # Convert OneHot labels to Integer labels for PyTorch CrossEntropy
        if y_train.shape[1] > 1:
            y_train_converted = y_train.argmax(1)
        else:
            y_train_converted = y_train.flatten()

        # Train: fit() uses self.batch_size initialized earlier
        model.fit(x_train, y_train_converted)
        
        # Predict/Score: uses self.batch_size
        score = model.score(x_test, y_test)
        scores.append(score)
        print(f"{score=:.4f}")

    print(f"GraphCS {args.species} {args.tissue} {args.test_dataset}:")
    print(f"{scores}\n{np.mean(scores):.5f} +/- {np.std(scores):.5f}")

"""To reproduce GraphCS benchmarks, please refer to command lines below:

Mouse Brain
$ python graphcs.py --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695 --lr 0.001 --hidden 256

Mouse Spleen
$ python graphcs.py --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759 --vat_lr 0.1

Mouse Kidney
$ python graphcs.py --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

$ python graphcs.py --species human --tissue Brain --train_dataset 328 --test_dataset 138
"""