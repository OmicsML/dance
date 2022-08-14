import os

import pandas as pd
from sklearn.svm import SVC


class SVM():
    """The SVM cell-type classification model.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace contains arguments of SVM. See parser help document for more info.
    prj_path: str
        project path

    """

    def __init__(self, args, prj_path="./"):
        self.args = args

    def fit(self, train_labels, train_cell_feat):
        """Train the classifier.

        Parameters
        ----------
        train_labels: np.array
            Training labels.
        train_cell_feat: np.array
            Training cell features.

        """

        self.train_labels, self.train_cell_feat = train_labels, train_cell_feat
        self.model = SVC(random_state=self.args.random_seed, probability=True)
        self.model.fit(self.train_cell_feat, self.train_labels)

    def predict(self, map_dict, id2label, test_label_dict, test_feat_dict, test_cell_id_dict):
        """Predict cell labels.

        Parameters
        ----------
        map_dict: dict
            The map dictionary.
        id2label: np.array
            The dictionary for converting ID to label.
        test_label_dict: dict
            The dictionary for labels in testing set.
        test_feat_dict: dict
            The dictionary for features in testing set.
        test_cell_id_dict: dict
            The dictionary for cell ids.

        Returns
        -------
        output: dict
            A diction of predicted celllabels.

        """
        self.map_dict = map_dict
        self.id2label = id2label
        self.test_label_dict = test_label_dict
        self.test_feat_dict = test_feat_dict
        self.test_cell_id_dict = test_cell_id_dict
        output = {}
        for num in self.args.test_dataset:
            score = self.model.predict_proba(self.test_feat_dict[num])  # [cell, class-num]
            pred_labels = []
            unsure_num = correct = 0
            for pred, t_label in zip(score, self.test_label_dict[num]):
                pred_label = self.id2label[pred.argmax().item()]
                if pred_label in self.map_dict[num][t_label]:
                    correct += 1
                pred_labels.append(pred_label)
            output[num] = pred_labels
        return output

    def score(self, map_dict, id2label, test_label_dict, test_feat_dict, test_cell_id_dict):
        """Model performance score measured by accuracy.

        Parameters
        ----------
        map_dict: dict
            The map dictionary.
        id2label: dict
            The dictionary for converting ID to label.
        test_label_dict: dict
            The dictionary for labels in testing set.
        test_feat_dict: dict
            The dictionary for features in testing set.
        test_cell_id_dict: dict
            The dictionary for cell ids.

        Returns
        -------
        accuracy_all: dict
            A diction of accuracy on different testing sets.

        """
        self.map_dict = map_dict
        self.id2label = id2label
        self.test_label_dict = test_label_dict
        self.test_feat_dict = test_feat_dict
        self.test_cell_id_dict = test_cell_id_dict
        accuracy_all = {}
        for num in self.args.test_dataset:
            score = self.model.predict_proba(self.test_feat_dict[num])  # [cell, class-num]
            pred_labels = []
            unsure_num = correct = 0
            for pred, t_label in zip(score, self.test_label_dict[num]):
                pred_label = self.id2label[pred.argmax().item()]
                if pred_label in self.map_dict[num][t_label]:
                    correct += 1
                pred_labels.append(pred_label)

            acc = correct / score.shape[0]
            print(f"SVM-{self.args.species}-{self.args.tissue}-{num}-ACC: {acc:.5f}")
            accuracy_all[num] = acc
        return accuracy_all

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
