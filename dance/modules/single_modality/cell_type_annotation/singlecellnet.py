import numpy as np
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier

from dance.modules.base import BaseClassificationMethod
from dance.transforms import AnnDataTransform, Compose, SCNFeature, SetConfig
from dance.typing import LogLevel, Optional


class SingleCellNet(BaseClassificationMethod):
    """The SingleCellNet model.

    Parameters
    ----------
    num_trees
        Number of trees in the random forest model.

    """

    def __init__(self, num_trees: int = 100):
        self.num_trees = num_trees

    @staticmethod
    def preprocessing_pipeline(normalize: bool = True, num_top_genes: int = 10, num_top_gene_pairs: int = 25,
                               log_level: LogLevel = "INFO"):
        transforms = []

        if normalize:
            transforms.append(AnnDataTransform(sc.pp.normalize_total, target_sum=1e4))
            transforms.append(AnnDataTransform(sc.pp.log1p))

        transforms.append(SCNFeature(num_top_genes=num_top_genes, num_top_gene_pairs=num_top_gene_pairs))
        transforms.append(SetConfig({"feature_channel": "SCNFeature", "label_channel": "cell_type"}))

        return Compose(*transforms, log_level=log_level)

    def randomize(self, exp, num: int = 50):
        """Return randomized features.

        Parameters
        ----------
        exp
            Data to be shuffled.
        num
            Number of samples to return.

        """
        rand = np.array([np.random.choice(x, len(x), replace=False) for x in exp]).T
        rand = np.array([np.random.choice(x, len(x), replace=False) for x in rand]).T
        return rand[:num]

    def fit(self, x, y, num_rand: int = 100, stratify: bool = True, random_state: Optional[int] = 100):
        """Train the SingleCellNet random forest model.

        Parameters
        ----------
        x
            Input features.
        y
            Labels.
        stratify
            Whether we select balanced class weight in the random forest model.
        random_state
            Random state.

        """
        x_rand = self.randomize(x, num=num_rand)
        x_comb = np.vstack((x, x_rand))

        y_rand = np.ones(x_rand.shape[0]) * (y.max() + 1)
        y_comb = np.concatenate((y, y_rand))

        self.model = RandomForestClassifier(n_estimators=self.num_trees, random_state=random_state,
                                            class_weight="balanced" if stratify else None)
        self.model.fit(x_comb, y_comb)

    def predict_proba(self, x):
        """Calculate predicted probabilities.

        Parameters
        ----------
        x
            Input featurex.

        Returns
        -------
        np.ndarray
            Cell-type probability matrix where each row is a cell and each column is a cell-type. The values in the
            matrix indicate the predicted probability that the cell is a particular cell-type. The last column
            corresponds to the probability that the model could not confidently identify the cell type of the cell.

        """
        return self.model.predict_proba(x)

    def predict(self, x):
        """Predict cell type label.

        Parameters
        ----------
        x
            Input features.

        Returns
        -------
        np.ndarray
            The most likely cell-type label of each sample.

        """
        pred_prob = self.predict_proba(x)
        pred = pred_prob.argmax(1)
        return pred
