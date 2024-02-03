import argparse
import pprint
from pathlib import Path
from typing import get_args

import wandb
from sklearn.random_projection import GaussianRandomProjection

from dance import logger
from dance.datasets.singlemodality import CellTypeAnnotationDataset
from dance.modules.single_modality.cell_type_annotation.svm import SVM
from dance.pipeline import PipelinePlaner, save_summary_data
from dance.registry import register_preprocessor
from dance.transforms.base import BaseTransform
from dance.typing import LogLevel
from dance.utils import set_seed


@register_preprocessor("feature", "cell")  # NOTE: register any custom preprocessing function to be used for tuning
class GaussRandProjFeature(BaseTransform):
    """Custom preprocessing to extract cell feature via Gaussian random projection."""

    _DISPLAY_ATTRS = ("n_components", "eps")

    def __init__(self, n_components: int = 400, eps: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_components = n_components
        self.eps = eps

    def __call__(self, data):
        feat = data.get_feature(return_type="numpy")
        grp = GaussianRandomProjection(n_components=self.n_components, eps=self.eps)

        self.logger.info(f"Start generateing cell feature via Gaussian random projection (d={self.n_components}).")
        data.data.obsm[self.out] = grp.fit_transform(feat)

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", action="store_true", help="Cache processed data.")
    parser.add_argument("--dense_dim", type=int, default=400, help="dim of PCA")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, set to -1 for CPU")
    parser.add_argument("--log_level", type=str, default="INFO", choices=get_args(LogLevel))
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--test_dataset", nargs="+", default=[2695], type=int, help="list of dataset id")
    parser.add_argument("--tissue", default="Brain")  # TODO: Add option for different tissue name for train/test
    parser.add_argument("--train_dataset", nargs="+", default=[753, 3285], type=int, help="list of dataset id")
    parser.add_argument("--tune_mode", default="pipeline", choices=["pipeline", "params"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--sweep_id", type=str, default=None)

    args = parser.parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"\n{pprint.pformat(vars(args))}")
    MAINDIR = Path(__file__).resolve().parent
    pipeline_planer = PipelinePlaner.from_config_file(f"{MAINDIR}/{args.tune_mode}_tuning_config.yaml")

    def evaluate_pipeline():
        wandb.init()

        set_seed(args.seed)
        model = SVM(args, random_state=args.seed)

        # Load raw data
        data = CellTypeAnnotationDataset(train_dataset=args.train_dataset, test_dataset=args.test_dataset,
                                         species=args.species, tissue=args.tissue).load_data()

        # Prepare preprocessing pipeline and apply it to data
        kwargs = {args.tune_mode: dict(wandb.config)}
        preprocessing_pipeline = pipeline_planer.generate(**kwargs)
        print(f"Pipeline config:\n{preprocessing_pipeline.to_yaml()}")
        preprocessing_pipeline(data)

        # Obtain training and testing data
        x_train, y_train = data.get_train_data()
        y_train_converted = y_train.argmax(1)  # convert one-hot representation into label index representation
        x_test, y_test = data.get_test_data()

        # Train and evaluate the model
        model.fit(x_train, y_train_converted)
        score = model.score(x_test, y_test)
        wandb.log({"acc": score})

        wandb.finish()

    entity, project, sweep_id = pipeline_planer.wandb_sweep_agent(evaluate_pipeline, sweep_id=args.sweep_id, count=3)

    # save_summary_data(entity,project,sweep_id,"temp.csv")
"""To reproduce SVM benchmarks, please refer to command lines below:

Mouse Brain
$ python main.py --tune_mode (pipeline/params) --species mouse --tissue Brain --train_dataset 753 3285 --test_dataset 2695

Mouse Spleen
$ python main.py --tune_mode (pipeline/params) --species mouse --tissue Spleen --train_dataset 1970 --test_dataset 1759

Mouse Kidney
$ python main.py --tune_mode (pipeline/params) --species mouse --tissue Kidney --train_dataset 4682 --test_dataset 203

"""
