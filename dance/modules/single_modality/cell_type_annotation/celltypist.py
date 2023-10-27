import os

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

from dance import logger
from dance.modules.base import BaseClassificationMethod
from dance.transforms import SetConfig
from dance.typing import LogLevel, Optional, Union


class Model():
    """Class that wraps the logistic Classifier and the StandardScaler.

    Parameters
    ----------
    clf:
        A logistic Classifier incorporated in the loaded model.
    scaler:
        A StandardScaler incorporated in the loaded model.
    description:
        Description of the model as a dictionary.

    Attributes
    ----------
    classifier:
        The logistic Classifier incorporated in the loaded model.
    scaler:
        The StandardScaler incorporated in the loaded model.
    description:
        Description of the loaded model.

    """

    def __init__(self, clf, scaler, description):
        self.classifier = clf
        self.scaler = scaler
        self.description = description

    @property
    def cell_types(self) -> np.ndarray:
        """Get cell types included in the model."""
        return self.classifier.classes_

    @property
    def features(self) -> np.ndarray:
        """Get genes included in the model."""
        return self.classifier.features

    def __repr__(self):
        """General the description."""
        base = f"CellTypist model with {len(self.cell_types)} cell types and {len(self.features)} features"
        for x in ['date', 'details', 'source', 'version']:
            if self.description[x] != '':
                base += f"\n    {x}: {self.description[x]}"
        if len(self.cell_types) == 2:
            base += f"\n    cell types: {self.cell_types[0]}, {self.cell_types[1]}"
            base += f"\n    features: {self.features[0]}, {self.features[1]}, ..., {self.features[-1]}"
        else:
            base += f"\n    cell types: {self.cell_types[0]}, {self.cell_types[1]}, ..., {self.cell_types[-1]}"
            base += "\n    features: {self.features[0]}, {self.features[1]}, ..., {self.features[-1]}"
        return base

    def predict_labels_and_prob(self, indata) -> tuple:
        """Get the decision matrix, probability matrix, and predicted cell types for the
        input data.

        Parameters
        ----------
        indata
            The input array-like object used as a query.

        Returns
        ----------
        tuple
            A tuple of decision score matrix, raw probability matrix, and predicted cell type labels.

        """
        scores = self.classifier.decision_function(indata)
        if len(self.cell_types) == 2:
            scores = np.column_stack([-scores, scores])
        probs = expit(scores)
        return scores, probs, self.classifier.classes_[scores.argmax(axis=1)]

    def extract_top_markers(self, cell_type: str, top_n: int = 10, only_positive: bool = True) -> np.ndarray:
        """Extract the top driving genes for a given cell type.

        Parameters
        ----------
        cell_type: str
            The cell type to extract markers for.
        top_n: int optional
            Number of markers to extract for a given cell type.
            (Default: 10)
        only_positive: bool optional
            Whether to extract positive markers only. Set to ``False`` to include negative markers as well.
            (Default: ``True``)

        Returns
        ----------
        :class:`~numpy.ndarray`
            A list of marker genes for the query cell type.

        """
        if cell_type not in self.cell_types:
            raise ValueError(f" '{cell_type}' is not found. Please provide a valid cell type name")
        if len(self.cell_types) == 2:
            coef_vector = self.classifier.coef_[0] if cell_type == self.cell_types[1] else -self.classifier.coef_[0]
        else:
            coef_vector = self.classifier.coef_[self.cell_types == cell_type][0]
        if not only_positive:
            coef_vector = np.abs(coef_vector)
        return self.features[np.argsort(-coef_vector)][:top_n]


# TODO: repurpose this to general purpose anlaysis tools
# (dance.tools? -> e.g., dance.tools.SummaryFrequency(data), how about preds?)
class AnnotationResult():
    """Class that represents the result of a celltyping annotation process.

    Parameters
    ----------
    labels
        A :class:`~pandas.DataFrame` object returned from the celltyping process, showing the predicted labels.
    decision_mat
        A :class:`~pandas.DataFrame` object returned from the celltyping process, showing the decision matrix.
    prob_mat
        A :class:`~pandas.DataFrame` object returned from the celltyping process, showing the probability matrix.
    adata
        An :class:`~anndata.AnnData` object representing the input object.

    Attributes
    ----------
    predicted_labels
        Predicted labels including the individual prediction results and (if majority voting is done) majority voting
        results.
    decision_matrix
        Decision matrix with the decision score of each cell belonging to a given cell type.
    probability_matrix
        Probability matrix representing the probability each cell belongs to a given cell type (transformed from
        decision matrix by the sigmoid function).
    cell_count
        Number of input cells which undergo the prediction process.
    adata
        An :class:`~anndata.AnnData` object representing the input data.

    """

    def __init__(self, labels: pd.DataFrame, decision_mat: pd.DataFrame, prob_mat: pd.DataFrame, adata: AnnData):
        self.predicted_labels = labels
        self.decision_matrix = decision_mat
        self.probability_matrix = prob_mat
        self.adata = adata
        self.cell_count = labels.shape[0]

    def summary_frequency(self, by: str = 'predicted_labels') -> pd.DataFrame:
        """
        Get the frequency of cells belonging to each cell type predicted by celltypist.
        Parameters

        ----------
        by: str
            Column name of :attr:`~celltypist.classifier.AnnotationResult.predicted_labels` specifying the prediction
            type which the summary is based on. Set to ``'majority_voting'`` if you want to summarize for the majority
            voting classifier.
            (Default: ``'predicted_labels'``)

        Returns
        ----------
        :class:`~pandas.DataFrame`
            A :class:`~pandas.DataFrame` object with cell type frequencies.
        """
        unique, counts = np.unique(self.predicted_labels[by], return_counts=True)
        df = pd.DataFrame(list(zip(unique, counts)), columns=["celltype", "counts"])
        df.sort_values(['counts'], ascending=False, inplace=True)
        return df

    def to_adata(self, insert_labels: bool = True, insert_conf: bool = True, insert_conf_by: str = 'predicted_labels',
                 insert_decision: bool = False, insert_prob: bool = False, prefix: str = '') -> AnnData:
        """Insert the predicted labels, decision or probability matrix, and (if majority
        voting is done) majority voting results into the AnnData object.

        Parameters
        ----------
        insert_labels: bool optional
            Whether to insert the predicted cell type labels and (if majority voting is done) majority voting-based
            labels into the AnnData object. (Default: ``True``)
        insert_conf: bool optional
            Whether to insert the confidence scores into the AnnData object. (Default: ``True``)
        insert_conf_by: str optional
            Column name of :attr:`~celltypist.classifier.AnnotationResult.predicted_labels` specifying the prediction
            type which the confidence scores are based on. Setting to ``'majority_voting'`` will insert the confidence
            scores corresponding to the majority-voting result.
            (Default: ``'predicted_labels'``)
        insert_decision: bool optional
            Whether to insert the decision matrix into the AnnData object. (Default: ``False``)
        insert_prob: bool optional
            Whether to insert the probability matrix into the AnnData object. This will override the decision matrix
            even when ``insert_decision`` is set to ``True``. (Default: ``False``)
        prefix:  str optional
            Prefix for the inserted columns in the AnnData object. Default to no prefix used.

        Returns
        ----------
        :class:`~anndata.AnnData`
            Depending on whether majority voting is done, an :class:`~anndata.AnnData` object with the following columns
            (prefixed with ``prefix``) added to the observation metadata:
            1) **predicted_labels**, individual prediction outcome for each cell.
            2) **over_clustering**, over-clustering result for the cells.
            3) **majority_voting**, the cell type label assigned to each cell after the majority voting process.
            4) **conf_score**, the confidence score of each cell.
            5) **name of each cell type**, which represents the decision scores (or probabilities if ``insert_prob`` is
               ``True``) of a given cell type across cells.

        """
        if insert_labels:
            self.adata.obs[[f"{prefix}{x}" for x in self.predicted_labels.columns]] = self.predicted_labels
        if insert_conf:
            if insert_conf_by == 'predicted_labels':
                self.adata.obs[f"{prefix}conf_score"] = self.probability_matrix.max(axis=1).values
            elif insert_conf_by == 'majority_voting':
                if insert_conf_by not in self.predicted_labels:
                    raise KeyError(" Did not find the column `majority_voting` in the "
                                   "`AnnotationResult.predicted_labels`, perform majority voting beforehand or use "
                                   "`insert_conf_by = 'predicted_labels'` instead")
                self.adata.obs[f"{prefix}conf_score"] = [
                    row[self.predicted_labels.majority_voting[index]]
                    for index, row in self.probability_matrix.iterrows()
                ]
            else:
                raise KeyError(f" Unrecognized `insert_conf_by` value ('{insert_conf_by}'), should be one of "
                               "`'predicted_labels'` or `'majority_voting'`")
        if insert_prob:
            self.adata.obs[[f"{prefix}{x}" for x in self.probability_matrix.columns]] = self.probability_matrix
        elif insert_decision:
            self.adata.obs[[f"{prefix}{x}" for x in self.decision_matrix.columns]] = self.decision_matrix
        return self.adata

    def to_plots(self, folder: str, plot_probability: bool = False, format: str = 'pdf', prefix: str = '') -> None:
        """Plot the celltyping and (if majority voting is done) majority-voting results.

        Parameters
        ----------
        folder: str
            Path to a folder which stores the output figures.
        plot_probability: bool optional
            Whether to also plot the decision score and probability distributions of each cell type across the test
            cells. If ``True``, a number of figures will be generated (may take some time if the input data is large).
            (Default: ``False``)
        format: str optional
            Format of output figures. Default to vector PDF files (note dots are still drawn with png backend).
            (Default: ``'pdf'``)
        prefix: str optional
            Prefix for the output figures. Default to no prefix used.

        Returns
        ----------
        None
            Depending on whether majority voting is done and ``plot_probability``, multiple UMAP plots showing the
            prediction and majority voting results in the ``folder``:
            1) **predicted_labels**, individual prediction outcome for each cell overlaid onto the UMAP.
            2) **over_clustering**, over-clustering result of the cells overlaid onto the UMAP.
            3) **majority_voting**, the cell type label assigned to each cell after the majority voting process overlaid
               onto the UMAP.
            4) **name of each cell type**, which represents the decision scores and probabilities of a given cell type
               distributed across cells overlaid onto the UMAP.

        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f" Output folder {folder} does not exist. Please provide a valid folder")
        if 'X_umap' in self.adata.obsm:
            logger.info("Detected existing UMAP coordinates, will plot the results accordingly")
        elif 'connectivities' in self.adata.obsp:
            logger.info(" Generating UMAP coordinates based on the neighborhood graph")
            sc.tl.umap(self.adata)
        else:
            logger.info("Constructing the neighborhood graph and generating UMAP coordinates")
            adata = self.adata.copy()
            self.adata.obsm['X_pca'], self.adata.obsp['connectivities'], self.adata.obsp['distances'], self.adata.uns[
                'neighbors'] = Classifier._construct_neighbor_graph(adata)
            sc.tl.umap(self.adata)
        logger.info("Plotting the results")
        sc.settings.set_figure_params(figsize=[6.4, 6.4], format=format)
        self.adata.obs[self.predicted_labels.columns] = self.predicted_labels
        for column in self.predicted_labels:
            sc.pl.umap(self.adata, color=column, legend_loc='on data', show=False, legend_fontweight='normal',
                       title=column.replace('_', ' '))
            plt.savefig(os.path.join(folder, prefix + column + '.' + format))
        if plot_probability:
            for column in self.probability_matrix:
                self.adata.obs['decision score'] = self.decision_matrix[column]
                self.adata.obs['probability'] = self.probability_matrix[column]
                sc.pl.umap(self.adata, color=['decision score', 'probability'], show=False)
                plt.savefig(os.path.join(folder, prefix + column.replace('/', '_') + '.' + format))
            self.adata.obs.drop(columns=['decision score', 'probability'], inplace=True)

    def to_table(self, folder: str, prefix: str = '', xlsx: bool = False) -> None:
        """Write out tables of predicted labels, decision matrix, and probability
        matrix.

        Parameters
        ----------
        folder: str
            Path to a folder which stores the output table/tables.
        prefix: str
            Prefix for the output table/tables. Default to no prefix used.
        xlsx: bool optional
            Whether to merge output tables into a single Excel (.xlsx).
            (Default: ``False``)

        Returns
        ----------
        None
            Depending on ``xlsx``, return table(s) of predicted labels, decision matrix and probability matrix.

        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f" Output folder {folder} does not exist. Please provide a valid folder")
        if not xlsx:
            self.predicted_labels.to_csv(os.path.join(folder, f"{prefix}predicted_labels.csv"))
            self.decision_matrix.to_csv(os.path.join(folder, f"{prefix}decision_matrix.csv"))
            self.probability_matrix.to_csv(os.path.join(folder, f"{prefix}probability_matrix.csv"))
        else:
            with pd.ExcelWriter(os.path.join(folder, f"{prefix}annotation_result.xlsx")) as writer:
                self.predicted_labels.to_excel(writer, sheet_name="Predicted Labels")
                self.decision_matrix.to_excel(writer, sheet_name="Decision Matrix")
                self.probability_matrix.to_excel(writer, sheet_name="Probability Matrix")

    def __repr__(self):
        base = f"CellTypist prediction result for {self.cell_count} query cells"
        base += f"\n    predicted_labels: data frame with {self.predicted_labels.shape[1]} "
        base += f"{'columns' if self.predicted_labels.shape[1] > 1 else 'column'} "
        base += f"({str(list(self.predicted_labels.columns))[1:-1]})"
        base += f"\n    decision_matrix: data frame with {self.cell_count} query cells and "
        base += f"{self.decision_matrix.shape[1]} cell types"
        base += f"\n    probability_matrix: data frame with {self.cell_count} query cells and "
        base += f"{self.probability_matrix.shape[1]} cell types"
        base += "\n    adata: AnnData object referred"
        return base


class Classifier():
    """Class that wraps the celltyping and majority voting processes.

    Parameters
    ----------
    x: np.ndarray
        Input expression matrix (cell x gene).
    model: Model
        A :class:`~celltypist.models.Model` object that wraps the logistic Classifier and the StandardScaler.

    Attributes
    ----------
    adata:
        An :class:`~anndata.AnnData` object which stores the log1p normalized expression data in ``.X`` or ``.raw.X``.
    indata:
        The expression matrix used for predictions stored in the log1p normalized format.
    indata_genes:
        All the genes included in the input data.
    indata_names:
        All the cells included in the input data.
    model:
        A :class:`~celltypist.models.Model` object that wraps the logistic Classifier and the StandardScaler.

    """

    def __init__(self, x: np.ndarray, model: Model):
        self.model = model

        self.adata = AnnData(x)
        self.adata.var_names_make_unique()

        self.indata = self.adata.X
        self.indata_genes = self.adata.var_names
        self.indata_names = self.adata.obs_names
        logger.info(f"Input data has {self.indata.shape[0]:,} cells and {len(self.indata_genes):,} genes")

    def celltype(self) -> AnnotationResult:
        """Run celltyping jobs to predict cell types of input data.

        Returns
        ----------
        :class:`~celltypist.classifier.AnnotationResult`
            An :class:`~celltypist.classifier.AnnotationResult` object. Four important attributes within this class are:
            1) :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`, predicted labels from celltypist.
            2) :attr:`~celltypist.classifier.AnnotationResult.decision_matrix`, decision matrix from celltypist.
            3) :attr:`~celltypist.classifier.AnnotationResult.probability_matrix`, probability matrix from celltypist.
            4) :attr:`~celltypist.classifier.AnnotationResult.adata`, AnnData object representation of the input data.

        """
        logger.info("Matching reference genes in the model")
        k_x = np.isin(self.indata_genes, self.model.classifier.features)
        if k_x.sum() == 0:
            raise ValueError("No features overlap with the model. Please provide gene symbols")
        else:
            logger.info(f"{k_x.sum():,} features used for prediction")
        k_x_idx = np.where(k_x)[0]
        self.indata_genes = self.indata_genes[k_x_idx]
        lr_idx = np.where(np.isin(self.model.classifier.features, self.indata_genes))[0]

        logger.info("Scaling input data")
        means_ = self.model.scaler.mean_[lr_idx]
        sds_ = self.model.scaler.scale_[lr_idx]
        self.indata = (self.indata[:, k_x_idx] - means_) / sds_
        self.indata[self.indata > 10] = 10

        # Temporarily replace with subsetted features, will recover after running the prediction function
        ni, fs, cf = self.model.classifier.n_features_in_, self.model.classifier.features, self.model.classifier.coef_
        self.model.classifier.n_features_in_ = lr_idx.size
        self.model.classifier.features = self.model.classifier.features[lr_idx]
        self.model.classifier.coef_ = self.model.classifier.coef_[:, lr_idx]

        logger.info("Predicting labels")
        decision_mat, prob_mat, lab = self.model.predict_labels_and_prob(self.indata)
        logger.info("Prediction done")

        # Restore model after prediction
        self.model.classifier.n_features_in_, self.model.classifier.features, self.model.classifier.coef_ = ni, fs, cf

        cells = self.indata_names
        return AnnotationResult(pd.DataFrame(lab, columns=['predicted_labels'], index=cells, dtype='category'),
                                pd.DataFrame(decision_mat, columns=self.model.classifier.classes_, index=cells),
                                pd.DataFrame(prob_mat, columns=self.model.classifier.classes_, index=cells), self.adata)

    @staticmethod
    def _construct_neighbor_graph(adata: AnnData) -> tuple:
        """Construct a neighborhood graph.

        This function is for internal use.

        """
        if 'X_pca' not in adata.obsm.keys():
            if adata.X.min() < 0:
                adata = adata.raw.to_adata()
            if 'highly_variable' not in adata.var:
                sc.pp.filter_genes(adata, min_cells=5)
                sc.pp.highly_variable_genes(adata, n_top_genes=min([2500, adata.n_vars]))
            adata = adata[:, adata.var.highly_variable]
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)
        return adata.obsm['X_pca'], adata.obsp['connectivities'], adata.obsp['distances'], adata.uns['neighbors']

    def over_cluster(self, resolution: Optional[float] = None) -> pd.Series:
        """Over-clustering input data with a canonical Scanpy pipeline. A neighborhood
        graph will be used (or constructed if not found) for the over-clustering.

        Parameters
        ----------
        resolution: float optional
            Resolution parameter for leiden clustering which controls the coarseness of the clustering.
            Default to 5, 10, 15, 20, 25 and 30 for datasets with cell numbers less than 5k, 20k, 40k, 100k, 200k and
            above, respectively.

        Returns
        ----------
        :class:`~pandas.Series`
            A :class:`~pandas.Series` object showing the over-clustering result.

        """
        if 'connectivities' not in self.adata.obsp:
            logger.info("Can not detect a neighborhood graph, will construct one before the over-clustering")
            adata = self.adata.copy()
            self.adata.obsm['X_pca'], self.adata.obsp['connectivities'], self.adata.obsp['distances'], self.adata.uns[
                'neighbors'] = Classifier._construct_neighbor_graph(adata)
        else:
            logger.info("Detected a neighborhood graph in the input object, will run overclustering on the basis of it")
        if resolution is None:
            if self.adata.n_obs < 5000:
                resolution = 5
            elif self.adata.n_obs < 20000:
                resolution = 10
            elif self.adata.n_obs < 40000:
                resolution = 15
            elif self.adata.n_obs < 100000:
                resolution = 20
            elif self.adata.n_obs < 200000:
                resolution = 25
            else:
                resolution = 30
        logger.info(f"Over-clustering input data with resolution set to {resolution}")
        sc.tl.leiden(self.adata, resolution=resolution, key_added='over_clustering')
        return self.adata.obs.pop('over_clustering')

    @staticmethod
    def majority_vote(predictions: AnnotationResult, over_clustering: Union[list, tuple, np.ndarray, pd.Series,
                                                                            pd.Index],
                      min_prop: float = 0) -> AnnotationResult:
        """Majority vote the celltypist predictions using the result from the over-
        clustering.

        Parameters
        ----------
        predictions: AnnotationResult
            An :class:`~celltypist.classifier.AnnotationResult` object containing the
            :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`.
        over_clustering: Union[list, tuple, np.ndarray, pd.Series, pd.Index]
            A list, tuple, numpy array, pandas series or index containing the over-clustering information.
        min_prop: float
            For the dominant cell type within a subcluster, the minimum proportion of cells required to support naming
            of the subcluster by this cell type. (Default: 0)

        Returns
        ----------
        output:class:`~celltypist.classifier.AnnotationResult`
            An :class:`~celltypist.classifier.AnnotationResult` object. Four important attributes within this class are:
            1) :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`, predicted labels from celltypist.
            2) :attr:`~celltypist.classifier.AnnotationResult.decision_matrix`, decision matrix from celltypist.
            3) :attr:`~celltypist.classifier.AnnotationResult.probability_matrix`, probability matrix from celltypist.
            4) :attr:`~celltypist.classifier.AnnotationResult.adata`, AnnData object representation of the input data.

        """
        if isinstance(over_clustering, (list, tuple)):
            over_clustering = np.array(over_clustering)
        logger.info(" Majority voting the predictions")
        votes = pd.crosstab(predictions.predicted_labels['predicted_labels'], over_clustering)
        majority = votes.idxmax(axis=0)
        freqs = (votes / votes.sum(axis=0).values).max(axis=0)
        majority[freqs < min_prop] = 'Heterogeneous'
        majority = majority[over_clustering].reset_index()
        majority.index = predictions.predicted_labels.index
        majority.columns = ['over_clustering', 'majority_voting']
        majority['majority_voting'] = majority['majority_voting'].astype('category')
        predictions.predicted_labels = predictions.predicted_labels.join(majority)
        logger.info(" Majority voting done!")
        return predictions


class Celltypist(BaseClassificationMethod):
    """The CellTypist cell annotation method.

    Parameters
    ----------
    majority_voting
        Whether to refine the predicted labels by running the majority voting classifier after over-clustering.

    """

    def __init__(self, majority_voting: bool = False, clf=None, scaler=None, description=None):
        self.majority_voting = majority_voting
        self.classifier = clf
        self.scaler = scaler
        self.description = description

    @staticmethod
    def preprocessing_pipeline(log_level: LogLevel = "INFO"):
        return SetConfig({"label_channel": "cell_type"}, log_level=log_level)

    def fit(self, indata: np.array, labels: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
            C: float = 1.0, solver: Optional[str] = None, max_iter: int = 1000, n_jobs: Optional[int] = None,
            use_SGD: bool = False, alpha: float = 0.0001, mini_batch: bool = False, batch_number: int = 100,
            batch_size: int = 1000, epochs: int = 10, balance_cell_type: bool = False, feature_selection: bool = False,
            top_genes: int = 300, **kwargs):
        """Train a celltypist model using mini-batch (optional) logistic classifier with
        a global solver or stochastic gradient descent (SGD) learning.

        Parameters
        ----------
        indata: np.ndarray
            Input gene expression matrix (cell x gene).
        labels: np.array
            1-D numpy array indicating cell-type identities of each cell (in index of the cell-types).
        C: float optional
            Inverse of L2 regularization strength for traditional logistic classifier. A smaller value can possibly
            improve model generalization while at the cost of decreased accuracy. This argument is ignored if SGD
            learning is enabled (``use_SGD = True``). (Default: 1.0)
        solver: str optional
            Algorithm to use in the optimization problem for traditional logistic classifier. The default behavior is
            to choose the solver according to the size of the input data. This argument is ignored if SGD learning is
            enabled (``use_SGD = True``).
        max_iter: int optional
            Maximum number of iterations before reaching the minimum of the cost function.
            Try to decrease ``max_iter`` if the cost function does not converge for a long time.
            This argument is for both traditional and SGD logistic classifiers, and will be ignored if mini-batch SGD
            training is conducted (``use_SGD = True`` and ``mini_batch = True``). (Default: 1000)
        n_jobs: int optional
            Number of CPUs used. Default to one CPU. ``-1`` means all CPUs are used.
            This argument is for both traditional and SGD logistic classifiers.
        use_SGD: bool optional
            Whether to implement SGD learning for the logistic classifier. (Default: ``False``)
        alpha: float optional
            L2 regularization strength for SGD logistic classifier. A larger value can possibly improve model
            generalization while at the cost of decreased accuracy. This argument is ignored if SGD learning is disabled
            (``use_SGD = False``). (Default: 0.0001)
        mini_batch: bool optional
            Whether to implement mini-batch training for the SGD logistic classifier.
            Setting to ``True`` may improve the training efficiency for large datasets (for example, >100k cells).
            This argument is ignored if SGD learning is disabled (``use_SGD = False``). (Default: ``False``)
        batch_number: int optional
            The number of batches used for training in each epoch. Each batch contains ``batch_size`` cells. For
            datasets which cannot be binned into ``batch_number`` batches, all batches will be used. This argument is
            relevant only if mini-batch SGD training is conducted (``use_SGD = True`` and ``mini_batch = True``).
            (Default: 100)
        batch_size: int optional
            The number of cells within each batch. This argument is relevant only if mini-batch SGD training is
            conducted (``use_SGD = True`` and ``mini_batch = True``). (Default: 1000)
        epochs: int optional
            The number of epochs for the mini-batch training procedure. The default values of ``batch_number``,
            ``batch_size``, and ``epochs`` together allow observing ~10^6 training cells. This argument is relevant
            only if mini-batch SGD training is conducted (``use_SGD = True`` and ``mini_batch = True``). (Default: 10)
        balance_cell_type: bool optional
            Whether to balance the cell type frequencies in mini-batches during each epoch. Setting to ``True`` will
            sample rare cell types with a higher probability, ensuring close-to-even cell type distributions in
            mini-batches. This argument is relevant only if mini-batch SGD training is conducted (``use_SGD = True`` and
            ``mini_batch = True``). (Default: ``False``)
        feature_selection: bool optional
            Whether to perform two-pass data training where the first round is used for selecting important
            features/genes using SGD learning. If ``True``, the training time will be longer. (Default: ``False``)
        top_genes: int optional
            The number of top genes selected from each class/cell-type based on their absolute regression coefficients.
            The final feature set is combined across all classes (i.e., union). (Default: 300)
        **kwargs
            Other keyword arguments passed to :class:`~sklearn.linear_model.LogisticRegression` (``use_SGD = False``) or
            :class:`~sklearn.linear_model.SGDClassifier` (``use_SGD = True``).

        Returns
        -------
        :class:`~celltypist.models.Model`
            An instance of the :class:`~celltypist.models.Model` trained by celltypist.

        """
        # Prepare
        logger.info("Preparing data before training")
        genes = np.arange(indata.shape[1]).astype(str)

        # Scaler
        logger.info("Scaling input data")
        scaler = StandardScaler()
        indata = scaler.fit_transform(indata)
        indata[indata > 10] = 10

        # Classifier
        if use_SGD or feature_selection:
            classifier = SGDClassifier_celltypist(indata=indata, labels=labels, alpha=alpha, max_iter=max_iter,
                                                  n_jobs=n_jobs, mini_batch=mini_batch, batch_number=batch_number,
                                                  batch_size=batch_size, epochs=epochs,
                                                  balance_cell_type=balance_cell_type, **kwargs)
        else:
            classifier = LRClassifier_celltypist(indata=indata, labels=labels, C=C, solver=solver, max_iter=max_iter,
                                                 n_jobs=n_jobs, **kwargs)

        # Feature selection -> new classifier and scaler
        if feature_selection:
            logger.info("Selecting features")
            if len(genes) <= top_genes:
                raise ValueError(f" The number of genes ({len(genes)}) is fewer than the `top_genes` ({top_genes}). "
                                 "Unable to perform feature selection")
            gene_index = np.argpartition(np.abs(classifier.coef_), -top_genes, axis=1)[:, -top_genes:]
            gene_index = np.unique(gene_index)
            logger.info(f"{len(gene_index)} features are selected")
            genes = genes[gene_index]

            logger.info("Starting the second round of training")
            if use_SGD:
                classifier = SGDClassifier_celltypist(indata=indata[:, gene_index], labels=labels, alpha=alpha,
                                                      max_iter=max_iter, n_jobs=n_jobs, mini_batch=mini_batch,
                                                      batch_number=batch_number, batch_size=batch_size, epochs=epochs,
                                                      balance_cell_type=balance_cell_type, **kwargs)
            else:
                classifier = LRClassifier_celltypist(indata=indata[:, gene_index], labels=labels, C=C, solver=solver,
                                                     max_iter=max_iter, n_jobs=n_jobs, **kwargs)
            scaler.mean_ = scaler.mean_[gene_index]
            scaler.var_ = scaler.var_[gene_index]
            scaler.scale_ = scaler.scale_[gene_index]
            scaler.n_features_in_ = len(gene_index)

        # Model finalization
        classifier.features = genes
        description = {'number_celltypes': len(classifier.classes_)}
        logger.info("Model training done")

        self.classifier = classifier
        self.scaler = scaler
        self.description = description

    def predict(self, x: np.ndarray, as_obj: bool = False, over_clustering: Optional[Union[str, list, tuple, np.ndarray,
                                                                                           pd.Series, pd.Index]] = None,
                min_prop: float = 0) -> Union[np.ndarray, AnnotationResult]:
        """Run the prediction and (optional) majority voting to annotate the input
        dataset.

        Parameters
        ----------
        x: np.ndarray
            Input expression matrix (cell x gene).
        as_obj: bool
            If set to ``True``, then return the prediction results are :class:`~AnnotationResult`. Otherwise, return the
            predicted cell-label indexes ad 1-d numpy array instead. (Default: ``False``)
        over_clustering: Union[str, list, tuple, np.ndarray, pd.Series, pd.Index] optional
            This argument can be provided in several ways:
            1) an input plain file with the over-clustering result of one cell per line.
            2) a string key specifying an existing metadata column in the AnnData (pre-created by the user).
            3) a python list, tuple, numpy array, pandas series or index representing the over-clustering result of the
               input cells.
            4) if none of the above is provided, will use a heuristic over-clustering approach according to the size of
               input data.
            Ignored if ``majority_voting`` is set to ``False``.
        min_prop: float optional
            For the dominant cell type within a subcluster, the minimum proportion of cells required to support naming
            of the subcluster by this cell type. Ignored if ``majority_voting`` is set to ``False``. Subcluster that
            fails to pass this proportion threshold will be assigned ``'Heterogeneous'``. (Default: 0)

        """
        # Construct classifier
        lr_classifier = Model(self.classifier, self.scaler, self.description)
        clf = Classifier(x=x, model=lr_classifier)

        # Predict
        predictions = clf.celltype()

        if self.majority_voting:
            if predictions.cell_count <= 50:
                logger.warn("The input number of cells ({predictions.cell_count}) is too few to conduct "
                            "proper over-clustering; no majority voting is performed")
            else:
                predictions = self._majority_voting(predictions, clf, over_clustering, min_prop)

        if not as_obj:
            col = "majority_voting" if self.majority_voting else "predicted_labels"
            predictions = np.array(predictions.predicted_labels[col].values)

        return predictions

    def _majority_voting(self, predictions, clf, over_clustering, min_prop) -> AnnotationResult:
        # Over clustering
        if over_clustering is None:
            over_clustering = clf.over_cluster()
            predictions.adata = clf.adata
        elif isinstance(over_clustering, str):
            if over_clustering in clf.adata.obs:
                over_clustering = clf.adata.obs[over_clustering]
            else:
                logger.info(f"Did not identify '{over_clustering}' as a cell metadata column, "
                            "assume it to be a plain text file")
                try:
                    with open(over_clustering) as f:
                        over_clustering = [x.strip() for x in f.readlines()]
                except Exception as e:
                    raise Exception(f" {e}")

        if len(over_clustering) != clf.adata.n_obs:
            raise ValueError(f"Length of `over_clustering` ({len(over_clustering)}) does not match "
                             f"the number of input cells ({clf.adata.n_obs})")

        # Majority voting
        return Classifier.majority_vote(predictions, over_clustering, min_prop=min_prop)


def LRClassifier_celltypist(indata, labels, C, solver, max_iter, n_jobs, **kwargs) -> LogisticRegression:
    """For internal use.

    Get the logistic Classifier.

    """
    no_cells = len(labels)
    if solver is None:
        solver = 'sag' if no_cells > 50000 else 'lbfgs'
    elif solver not in ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'):
        raise ValueError("?? Invalid `solver`, should be one of `'liblinear'`, `'lbfgs'`, `'newton-cg'`, "
                         "`'sag'`, and `'saga'`")
    logger.info("Training data using logistic regression")
    if (no_cells > 100000) and (indata.shape[1] > 10000):
        logger.warn(f"?? Warning: it may take a long time to train this dataset with {no_cells} cells and "
                    f"{indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)")
    logger.info("LRClassifier training start...")
    classifier = LogisticRegression(C=C, solver=solver, max_iter=max_iter, multi_class='ovr', n_jobs=n_jobs, **kwargs)
    classifier.fit(indata, labels)
    return classifier


def SGDClassifier_celltypist(indata, labels, alpha, max_iter, n_jobs, mini_batch, batch_number, batch_size, epochs,
                             balance_cell_type, **kwargs) -> SGDClassifier:
    """For internal use.

    Get the SGDClassifier.

    """
    classifier = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=max_iter, n_jobs=n_jobs, **kwargs)
    if not mini_batch:
        logger.info("Training data using SGD logistic regression")
        if (len(labels) > 100000) and (indata.shape[1] > 10000):
            logger.warn(f"?? Warning: it may take a long time to train this dataset with {len(labels)} cells and "
                        f"{indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset "
                        "(e.g., hvgs)")
        logger.info("SGDlassifier training start...")
        classifier.fit(indata, labels)
    else:
        logger.info("Training data using mini-batch SGD logistic regression")
        no_cells = len(labels)
        if no_cells < 10000:
            logger.warn(f"?? Warning: the number of cells ({no_cells}) is not big enough to conduct a proper "
                        "mini-batch training. You may consider using traditional SGD classifier (mini_batch = False)")
        if no_cells <= batch_size:
            raise ValueError(f"?? Number of cells ({no_cells}) is fewer than the batch size ({batch_size}). Decrease "
                             "`batch_size`, or use SGD directly (mini_batch = False)")
        no_cells_sample = min([batch_number * batch_size, no_cells])
        starts = np.arange(0, no_cells_sample, batch_size)
        if balance_cell_type:
            celltype_freq = np.unique(labels, return_counts=True)
            len_celltype = len(celltype_freq[0])
            mapping = pd.Series(1 / (celltype_freq[1] * len_celltype), index=celltype_freq[0])
            p = mapping[labels].values
        for epoch in range(1, (epochs + 1)):
            logger.info(f"Epochs: [{epoch}/{epochs}]")
            if not balance_cell_type:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace=False)
            else:
                sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace=False, p=p)
            for start in starts:
                logger.info("SGDlassifier training start...")
                classifier.partial_fit(indata[sampled_cell_index[start:start + batch_size]],
                                       labels[sampled_cell_index[start:start + batch_size]], classes=np.unique(labels))
    return classifier
