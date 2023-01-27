import os

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from dance.modules.base import BaseClassificationMethod
from dance.transforms import Compose, SetConfig, WeightedFeaturePCA
from dance.typing import LogLevel, Optional
from dance.utils.deprecate import deprecated


class SVM(BaseClassificationMethod):
    """The SVM cell-type classification model.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace contains arguments of SVM. See parser help document for more info.
    prj_path: str
        project path

    """

    def __init__(self, args, prj_path="./", random_state: Optional[int] = None):
        self.args = args
        self.random_state = random_state
        self._mdl = SVC(random_state=random_state, probability=True)

    @staticmethod
    def preprocessing_pipeline(n_components: int = 400, log_level: LogLevel = "INFO"):
        return Compose(
            WeightedFeaturePCA(n_components=n_components, split_name="train"),
            SetConfig({
                "feature_channel": "WeightedFeaturePCA",
                "label_channel": "cell_type"
            }),
            log_level=log_level,
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Train the classifier.

        Parameters
        ----------
        x
            Training cell features.
        y
            Training labels.

        """
        self._mdl.fit(x, y)

    def predict(self, x: np.ndarray):
        """Predict cell labels.

        Parameters
        ----------
        x
            Samples to be predicted (samplex x features).

        Returns
        -------
        y
            Predicted labels of the input samples.

        """
        return self._mdl.predict(x)

    @deprecated
    def save(self, num, pred):
        """Save the predictions.

        Parameters
        ----------
        num: int
            test file name
        pred: dict
            prediction labels

        """
        label_map = pd.read_excel(self.prj_path / "data" / "celltype2subtype.xlsx", sheet_name=self.args.species,
                                  header=0, names=["species", "old_type", "new_type", "new_subtype"])

        save_path = self.prj_path / self.args.save_dir
        if not save_path.exists():
            save_path.mkdir()

        label_map = label_map.fillna("N/A", inplace=False)
        oldtype2newtype = {}
        oldtype2newsubtype = {}
        for _, old_type, new_type, new_subtype in label_map.itertuples(index=False):
            oldtype2newtype[old_type] = new_type
            oldtype2newsubtype[old_type] = new_subtype
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        df = pd.DataFrame({
            "index": self.test_cell_id_dict[num],
            "original label": self.test_label_dict[num],
            "cell type": [oldtype2newtype.get(p, p) for p in pred],
            "cell subtype": [oldtype2newsubtype.get(p, p) for p in pred]
        })
        df.to_csv(save_path / ("SVM_" + self.args.species + f"_{self.args.tissue}_{num}.csv"), index=False)
        print(f"output has been stored in {self.args.species}_{self.args.tissue}_{num}.csv")
