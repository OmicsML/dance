import json
import logging
import os
import pathlib
import pickle
import warnings
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd
import requests
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
from scipy.sparse import spmatrix
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dance.transforms.preprocess import (LRClassifier_celltypist, SGDClassifier_celltypist, downsample_adata,
                                         get_sample_csv_celltypist, get_sample_data_celltypist, prepare_data_celltypist,
                                         to_array_celltypist, to_vector_celltypist)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
set_level = logger.setLevel
info = logger.info
warn = logger.warning
error = logger.error
debug = logger.debug

celltypist_path = os.getenv('CELLTYPIST_FOLDER', default=os.path.join(str(pathlib.Path.home()), '.celltypist'))
pathlib.Path(celltypist_path).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(celltypist_path, "data")
models_path = os.path.join(data_path, "models")
pathlib.Path(models_path).mkdir(parents=True, exist_ok=True)

_samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples")


def _collapse_mean(arr: np.ndarray) -> Union[float, np.ndarray]:
    """For internal use.

    Average 1D array, or 2D array by row.

    """
    return np.mean(arr, axis=-1)


def _collapse_random(arr: np.ndarray) -> Union[float, np.ndarray]:
    """For internal use.

    Choose a random number from 1D array, or a random column from 2D array.

    """
    return np.random.choice(arr, 1)[0] if arr.ndim == 1 else arr[:, np.random.choice(arr.shape[1], 1)[0]]


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

    @staticmethod
    def load(self, model: Optional[str] = None):
        """Load the desired model.

        Parameters
        ----------
        model: Optional[str]
            Model name specifying the model you want to load. Default to `'Immune_All_Low.pkl'` if not provided.
            To see all available models and their descriptions, use :func:`~celltypist.models.models_description`.

        Returns
        ----------
        :class:`~celltypist.models.Model`
            A :class:`~celltypist.models.Model` object.

        """
        if not model:
            model = get_default_model()
        if model in get_all_models():
            model = get_model_path(model)
        if not os.path.isfile(model):
            raise FileNotFoundError(f" No such file: {model}")
        with open(model, "rb") as fh:
            try:
                pkl_obj = pickle.load(fh)
                return Model(pkl_obj['Model'], pkl_obj['Scaler_'], pkl_obj['description'])
            except Exception as exception:
                raise Exception(f" Invalid model: {model}. {exception}")

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
            base += f"\n    cell types: {self.cell_types[0]}, {self.cell_types[1]}\n    features: {self.features[0]}, {self.features[1]}, ..., {self.features[-1]}"
        else:
            base += f"\n    cell types: {self.cell_types[0]}, {self.cell_types[1]}, ..., {self.cell_types[-1]}\n    features: {self.features[0]}, {self.features[1]}, ..., {self.features[-1]}"
        return base

    def predict_labels_and_prob(self, indata, mode: str = 'best match', p_thres: float = 0.5) -> tuple:
        """Get the decision matrix, probability matrix, and predicted cell types for the
        input data.

        Parameters
        ----------
        indata
            The input array-like object used as a query.
        mode: str
            The way cell prediction is performed.
            For each query cell, the default (`'best match'`) is to choose the cell type with the largest score/probability as the final prediction.
            Setting to `'prob match'` will enable a multi-label classification, which assigns 0 (i.e., unassigned), 1, or >=2 cell type labels to each query cell.
            (Default: `'best match'`)
        p_thres: float
            Probability threshold for the multi-label classification. Ignored if `mode` is `'best match'`.
            (Default: 0.5)

        Returns
        ----------
        tuple
            A tuple of decision score matrix, raw probability matrix, and predicted cell type labels.

        """
        scores = self.classifier.decision_function(indata)
        if len(self.cell_types) == 2:
            scores = np.column_stack([-scores, scores])
        probs = expit(scores)
        if mode == 'best match':
            return scores, probs, self.classifier.classes_[scores.argmax(axis=1)]
        elif mode == 'prob match':
            flags = probs > p_thres
            labs = np.array(['|'.join(self.classifier.classes_[np.where(x)[0]]) for x in flags])
            labs[labs == ''] = 'Unassigned'
            return scores, probs, labs
        else:
            raise ValueError(f" Unrecognized `mode` value, should be one of `'best match'` or `'prob match'`")

    def write(self, file: str) -> None:
        """Write out the model."""
        obj = dict(Model=self.classifier, Scaler_=self.scaler, description=self.description)
        file = os.path.splitext(file)[0] + '.pkl'
        with open(file, 'wb') as output:
            pickle.dump(obj, output)

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
            Whether to extract positive markers only. Set to `False` to include negative markers as well.
            (Default: `True`)

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

    def convert(self, map_file: Optional[str] = None, sep: str = ',', convert_from: Optional[int] = None,
                convert_to: Optional[int] = None, unique_only: bool = True, collapse: str = 'average',
                random_state: int = 0) -> None:
        """Convert the model of one species to another species by mapping orthologous
        genes.

        Parameters
        ----------
        map_file: str optional
            A two-column gene mapping file between two species.
            Default to a human-mouse (mouse-human) conversion using the built-in mapping file provided by CellTypist.
        sep: str
            Delimiter of the mapping file. Default to comma (i.e., a csv file is by default expected from the user if provided).
        convert_from: int optional
            Column index (0 or 1) of the mapping file corresponding to the species converted from.
            Default to an automatic detection.
        convert_to: int optional
            Column index (0 or 1) of the mapping file corresponding to the species converted to.
            Default to an automatic detection.
        unique_only: bool  optional
            Whether to leverage only 1:1 orthologs between the two species.
            (Default: `True`)
        collapse: str optional
            The way 1:N orthologs are handled. Possible values are `'average'` which averages the classifier weights and `'random'` which randomly chooses one gene's weights from all its orthologs.
            This argument is ignored if `unique_only = True`.
            (Default: `'average'`)
        random_state: int optional
            Random seed for reproducibility. This argument is only relevant if `unique_only = False` and `collapse = 'random'`.

        Returns
        ----------
        None
            The original model is modified by converting to the other species.

        """
        map_file = get_sample_data_celltypist('Ensembl105_Human2Mouse_Genes.csv') if map_file is None else map_file
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f" No such file: {map_file}")
        #with and without headers are both ok -> real headers become fake genes and are removed afterwards
        map_content = pd.read_csv(map_file, sep=sep, header=None)
        map_content.dropna(axis=0, inplace=True)
        map_content.drop_duplicates(inplace=True)
        #From & To detection
        if (convert_from is None) and (convert_to is None):
            column1_overlap = map_content[0].isin(self.features).sum()
            column2_overlap = map_content[1].isin(self.features).sum()
            convert_from = 0 if column1_overlap > column2_overlap else 1
            convert_to = 1 - convert_from
        elif convert_from is None:
            if convert_to not in [0, 1]:
                raise ValueError(f" `convert_to` should be either 0 or 1")
            convert_from = 1 - convert_to
        elif convert_to is None:
            if convert_from not in [0, 1]:
                raise ValueError(f" `convert_from` should be either 0 or 1")
            convert_to = 1 - convert_from
        else:
            if {convert_from, convert_to} != {0, 1}:
                raise ValueError(f" `convert_from` and `convert_to` should be 0 (or 1) and 1 (or 0)")
        #filter
        map_content = map_content[map_content[convert_from].isin(self.features)]
        if unique_only:
            map_content.drop_duplicates([0], inplace=True)
            map_content.drop_duplicates([1], inplace=True)
        map_content['index_from'] = pd.DataFrame(
            self.features, columns=['features']).reset_index().set_index('features').loc[map_content[convert_from],
                                                                                         'index'].values
        #main
        logger.info(f" Number of genes in the original model: {len(self.features)}")
        features_to = map_content[convert_to].values if unique_only else np.unique(map_content[convert_to])
        if unique_only:
            index_from = map_content['index_from'].values
            self.classifier.coef_ = self.classifier.coef_[:, index_from]
            self.scaler.mean_ = self.scaler.mean_[index_from]
            self.scaler.var_ = self.scaler.var_[index_from]
            self.scaler.scale_ = self.scaler.scale_[index_from]
        else:
            if collapse not in ['average', 'random']:
                raise ValueError(f" Unrecognized `collapse` value, should be one of `'average'` or `'random'`")
            if collapse == 'random':
                np.random.seed(random_state)
            collapse_func = _collapse_mean if collapse == 'average' else _collapse_random
            coef_to = []
            mean_to = []
            var_to = []
            scale_to = []
            for feature_to in features_to:
                index_from = map_content[map_content[convert_to] == feature_to].index_from.values
                if len(index_from) == 1:
                    coef_to.append(self.classifier.coef_[:, index_from[0]])
                    mean_to.append(self.scaler.mean_[index_from[0]])
                    var_to.append(self.scaler.var_[index_from[0]])
                    scale_to.append(self.scaler.scale_[index_from[0]])
                else:
                    coef_to.append(collapse_func(self.classifier.coef_[:, index_from]))
                    mean_to.append(collapse_func(self.scaler.mean_[index_from]))
                    var_to.append(collapse_func(self.scaler.var_[index_from]))
                    scale_to.append(collapse_func(self.scaler.scale_[index_from]))
            self.classifier.coef_ = np.column_stack(coef_to)
            self.scaler.mean_ = np.array(mean_to)
            self.scaler.var_ = np.array(var_to)
            self.scaler.scale_ = np.array(scale_to)
        self.classifier.n_features_in_ = len(features_to)
        self.classifier.features = features_to
        self.scaler.n_features_in_ = len(features_to)
        logger.info(f" Conversion done! Number of genes in the converted model: {len(features_to)}")


def get_model_path(file: str) -> str:
    """Get the full path to a file in the `models` folder.

    Parameters
    ----------
    file: str
        File name as a string.
        To see all available models and their descriptions, use :func:`~celltypist.models.models_description`.

    Returns
    ----------
    str
        A string of the full path to the desired file.

    """
    return os.path.join(models_path, f"{file}")


def get_default_model() -> str:
    """Get the default model name.

    Returns
    ----------
    str
        A string showing the default model name (should be `'Immune_All_Low.pkl'`).

    """
    models_json = get_models_index()
    default_model = [m["filename"] for m in models_json["models"] if ("default" in m and m["default"])]
    if not default_model:
        first_model = models_json["models"][0]["filename"]
        logger.warn(f" No model marked as 'default', using {first_model}")
        return first_model
    if len(default_model) > 1:
        logger.warn(f" More than one model marked as 'default', using {default_model[0]}")
    return default_model[0]


def get_all_models() -> list:
    """
    Get a list of all the available models.
    Returns
    ----------
    list
        A list of available models.
    """
    download_if_required()
    available_models = []
    for model_filename in os.listdir(models_path):
        if model_filename.endswith(".pkl"):
            model_name = os.path.basename(model_filename)
            available_models.append(model_name)
    return available_models


def download_if_required() -> None:
    """Download models if there are none present in the `models` directory."""
    if len([m for m in os.listdir(models_path) if m.endswith(".pkl")]) == 0:
        logger.info(f" No available models. Downloading...")
        download_models()


def get_models_index(force_update: bool = False) -> dict:
    """Get the model json object containing the model list.

    Parameters
    ----------
    force_update: bool optional
        If set to `True`, will download the latest model json file from the remote.
        (Default: `False`)

    Returns
    ----------
    dict: dict
        A dict object converted from the model json file.

    """
    models_json_path = get_model_path("models.json")
    if not os.path.exists(models_json_path) or force_update:
        download_model_index()
    with open(models_json_path) as f:
        return json.load(f)


def download_model_index(only_model: bool = True) -> None:
    """Download the `models.json` file from the remote server.

    Parameters
    ----------
    only_model: bool
        If set to `False`, will also download the models in addition to the json file.
        (Default: `True`)

    Returns
    ----------
    No return

    """
    url = 'https://celltypist.cog.sanger.ac.uk/models/models.json'
    logger.info(f" Retrieving model list from server {url}")
    with open(get_model_path("models.json"), "wb") as f:
        f.write(requests.get(url).content)
    model_count = len(requests.get(url).json()["models"])
    logger.info(f" Total models in list: {model_count}")
    if not only_model:
        download_models()


def download_models(force_update: bool = False, model: Optional[Union[str, list, tuple]] = None) -> None:
    """Download all the available or selected models.

    Parameters
    ----------
    force_update: bool
        Whether to fetch a latest JSON index for downloading all available or selected models.
        Set to `True` if you want to parallel the latest celltypist model releases.
        (Default: `False`)
    model: Optional[Union[str, list, tuple]]
        Specific model(s) to download. By default, all available models are downloaded.
        Set to a specific model name or a list of model names to only download a subset of models.
        For example, set to `["ModelA.pkl", "ModelB.pkl"]` to only download ModelA and ModelB.
        To check all available models, use :func:`~celltypist.models.models_description`.

    """
    models_json = get_models_index(force_update)
    logger.info(f" Storing models in {models_path}")
    if model is not None:
        model_list = {model} if isinstance(model, str) else set(model)
        models_json["models"] = [m for m in models_json["models"] if m["filename"] in model_list]
        provided_no = len(model_list)
        filtered_no = len(models_json["models"])
        if filtered_no == 0:
            raise ValueError(f" No models match the celltypist model repertoire. Please provide valid model names")
        elif provided_no == filtered_no:
            logger.info(f" Total models to download: {provided_no}")
        else:
            ignored_models = model_list.difference({m["filename"] for m in models_json["models"]})
            logger.warn(
                f" Total models to download: {filtered_no}. {len(ignored_models)} not available: {ignored_models}")
    model_count = len(models_json["models"])
    for idx, model in enumerate(models_json["models"]):
        model_path = get_model_path(model["filename"])
        if os.path.exists(model_path) and not force_update:
            logger.info(f" Skipping [{idx+1}/{model_count}]: {model['filename']} (file exists)")
            continue
        logger.info(f" Downloading model [{idx+1}/{model_count}]: {model['filename']}")
        try:
            with open(model_path, "wb") as f:
                f.write(requests.get(model["url"]).content)
        except Exception as exception:
            logger.error(f" {model['filename']} failed {exception}")


def models_description(on_the_fly: bool = False) -> pd.DataFrame:
    """Get the descriptions of all available models.

    Parameters
    ----------
    on_the_fly: bool
        Whether to fetch the model information from downloaded model files.
        If set to `True`, will fetch the information by loading downloaded models.
        Default to fetching the information for all available models from the JSON file.
        (Default: `False`)

    Returns
    ----------
    :class:`~pandas.DataFrame`
        A :class:`~pandas.DataFrame` object with model descriptions.

    """
    logger.info(f" Detailed model information can be found at `https://www.celltypist.org/models`")
    if on_the_fly:
        filenames = get_all_models()
        descriptions = [Model.load(filename).description['details'] for filename in filenames]
    else:
        models_json = get_models_index()
        models = models_json["models"]
        filenames = [model['filename'] for model in models]
        descriptions = [model['details'] for model in models]
    return pd.DataFrame({'model': filenames, 'description': descriptions})


class AnnotationResult():
    """
    Class that represents the result of a celltyping annotation process.
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
        Predicted labels including the individual prediction results and (if majority voting is done) majority voting results.
    decision_matrix
        Decision matrix with the decision score of each cell belonging to a given cell type.
    probability_matrix
        Probability matrix representing the probability each cell belongs to a given cell type (transformed from decision matrix by the sigmoid function).
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
            Column name of :attr:`~celltypist.classifier.AnnotationResult.predicted_labels` specifying the prediction type which the summary is based on.
            Set to `'majority_voting'` if you want to summarize for the majority voting classifier.
            (Default: `'predicted_labels'`)

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
            Whether to insert the predicted cell type labels and (if majority voting is done) majority voting-based labels into the AnnData object.
            (Default: `True`)
        insert_conf: bool optional
            Whether to insert the confidence scores into the AnnData object.
            (Default: `True`)
        insert_conf_by: str optional
            Column name of :attr:`~celltypist.classifier.AnnotationResult.predicted_labels` specifying the prediction type which the confidence scores are based on.
            Setting to `'majority_voting'` will insert the confidence scores corresponding to the majority-voting result.
            (Default: `'predicted_labels'`)
        insert_decision: bool optional
            Whether to insert the decision matrix into the AnnData object.
            (Default: `False`)
        insert_prob: bool optional
            Whether to insert the probability matrix into the AnnData object. This will override the decision matrix even when `insert_decision` is set to `True`.
            (Default: `False`)
        prefix:  str optional
            Prefix for the inserted columns in the AnnData object. Default to no prefix used.

        Returns
        ----------
        :class:`~anndata.AnnData`
            Depending on whether majority voting is done, an :class:`~anndata.AnnData` object with the following columns (prefixed with `prefix`) added to the observation metadata:
            1) **predicted_labels**, individual prediction outcome for each cell.
            2) **over_clustering**, over-clustering result for the cells.
            3) **majority_voting**, the cell type label assigned to each cell after the majority voting process.
            4) **conf_score**, the confidence score of each cell.
            5) **name of each cell type**, which represents the decision scores (or probabilities if `insert_prob` is `True`) of a given cell type across cells.

        """
        if insert_labels:
            self.adata.obs[[f"{prefix}{x}" for x in self.predicted_labels.columns]] = self.predicted_labels
        if insert_conf:
            if insert_conf_by == 'predicted_labels':
                self.adata.obs[f"{prefix}conf_score"] = self.probability_matrix.max(axis=1).values
            elif insert_conf_by == 'majority_voting':
                if insert_conf_by not in self.predicted_labels:
                    raise KeyError(
                        f" Did not find the column `majority_voting` in the `AnnotationResult.predicted_labels`, perform majority voting beforehand or use `insert_conf_by = 'predicted_labels'` instead"
                    )
                self.adata.obs[f"{prefix}conf_score"] = [
                    row[self.predicted_labels.majority_voting[index]]
                    for index, row in self.probability_matrix.iterrows()
                ]
            else:
                raise KeyError(
                    f" Unrecognized `insert_conf_by` value ('{insert_conf_by}'), should be one of `'predicted_labels'` or `'majority_voting'`"
                )
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
            Whether to also plot the decision score and probability distributions of each cell type across the test cells.
            If `True`, a number of figures will be generated (may take some time if the input data is large).
            (Default: `False`)
        format: str optional
            Format of output figures. Default to vector PDF files (note dots are still drawn with png backend).
            (Default: `'pdf'`)
        prefix: str optional
            Prefix for the output figures. Default to no prefix used.

        Returns
        ----------
        None
            Depending on whether majority voting is done and `plot_probability`, multiple UMAP plots showing the prediction and majority voting results in the `folder`:
            1) **predicted_labels**, individual prediction outcome for each cell overlaid onto the UMAP.
            2) **over_clustering**, over-clustering result of the cells overlaid onto the UMAP.
            3) **majority_voting**, the cell type label assigned to each cell after the majority voting process overlaid onto the UMAP.
            4) **name of each cell type**, which represents the decision scores and probabilities of a given cell type distributed across cells overlaid onto the UMAP.

        """
        if not os.path.isdir(folder):
            raise FileNotFoundError(f" Output folder {folder} does not exist. Please provide a valid folder")
        if 'X_umap' in self.adata.obsm:
            logger.info(" Detected existing UMAP coordinates, will plot the results accordingly")
        elif 'connectivities' in self.adata.obsp:
            logger.info(" Generating UMAP coordinates based on the neighborhood graph")
            sc.tl.umap(self.adata)
        else:
            logger.info(" Constructing the neighborhood graph and generating UMAP coordinates")
            adata = self.adata.copy()
            self.adata.obsm['X_pca'], self.adata.obsp['connectivities'], self.adata.obsp['distances'], self.adata.uns[
                'neighbors'] = Classifier._construct_neighbor_graph(adata)
            sc.tl.umap(self.adata)
        logger.info(" Plotting the results")
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
            (Default: `False`)

        Returns
        ----------
        None
            Depending on `xlsx`, return table(s) of predicted labels, decision matrix and probability matrix.

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
        base += f"\n    predicted_labels: data frame with {self.predicted_labels.shape[1]} {'columns' if self.predicted_labels.shape[1] > 1 else 'column'} ({str(list(self.predicted_labels.columns))[1:-1]})"
        base += f"\n    decision_matrix: data frame with {self.cell_count} query cells and {self.decision_matrix.shape[1]} cell types"
        base += f"\n    probability_matrix: data frame with {self.cell_count} query cells and {self.probability_matrix.shape[1]} cell types"
        base += f"\n    adata: AnnData object referred"
        return base


class Classifier():
    """
    Class that wraps the celltyping and majority voting processes.
    Parameters
    ----------
    filename: Union[AnnData,str]
        Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData object (h5ad).
        If it's the former, a cell-by-gene format is desirable (see `transpose` for more information).
        Also accepts the input as an :class:`~anndata.AnnData` object already loaded in memory.
        Genes should be gene symbols. Non-expressed genes are preferred to be provided as well.
    model: Union[Model,str]
        A :class:`~celltypist.models.Model` object that wraps the logistic Classifier and the StandardScaler, the
        path to the desired model file, or the model name.
    transpose: bool
        Whether to transpose the input matrix. Set to `True` if `filename` is provided in a gene-by-cell format.
        (Default: `False`)
    gene_file: Optional[str]
        Path to the file which stores each gene per line corresponding to the genes used in the provided mtx file.
        Ignored if `filename` is not provided in the mtx format.
    cell_file: Optional[str]
        Path to the file which stores each cell per line corresponding to the cells used in the provided mtx file.
        Ignored if `filename` is not provided in the mtx format.
    Attributes
    ----------
    filename:
        Path to the input dataset. This attribute exists only when the input is a file path.
    adata:
        An :class:`~anndata.AnnData` object which stores the log1p normalized expression data in `.X` or `.raw.X`.
    indata:
        The expression matrix used for predictions stored in the log1p normalized format.
    indata_genes:
        All the genes included in the input data.
    indata_names:
        All the cells included in the input data.
    model:
        A :class:`~celltypist.models.Model` object that wraps the logistic Classifier and the StandardScaler.
    """

    def __init__(
        self,
        filename: Union[AnnData, str] = "",
        model: Union[Model, str] = "",
        transpose: bool = False,
        gene_file: Optional[str] = None,
        cell_file: Optional[str] = None,
        check_expression: bool = False,
    ):
        if isinstance(model, str):
            model = Model.load(model)
        self.model = model
        if not filename:
            logger.warn(f" No input file provided to the classifier")
            return
        if isinstance(filename, str):
            self.filename = filename
            logger.info(f" Input file is '{self.filename}'")
            logger.info(f" Loading data")
        if isinstance(filename, str) and filename.endswith(('.csv', '.txt', '.tsv', '.tab', '.mtx', '.mtx.gz')):
            self.adata = sc.read(self.filename)
            if transpose:
                self.adata = self.adata.transpose()
            if self.filename.endswith(('.mtx', '.mtx.gz')):
                if (gene_file is None) or (cell_file is None):
                    raise FileNotFoundError(
                        " Missing `gene_file` and/or `cell_file`. Please provide both arguments together with the input mtx file"
                    )
                genes_mtx = pd.read_csv(gene_file, header=None)[0].values
                cells_mtx = pd.read_csv(cell_file, header=None)[0].values
                if len(genes_mtx) != self.adata.n_vars:
                    raise ValueError(
                        f" The number of genes in {gene_file} does not match the number of genes in {self.filename}")
                if len(cells_mtx) != self.adata.n_obs:
                    raise ValueError(
                        f" The number of cells in {cell_file} does not match the number of cells in {self.filename}")
                self.adata.var_names = genes_mtx
                self.adata.obs_names = cells_mtx
            self.adata.var_names_make_unique()
            if not float(self.adata.X.max()).is_integer():
                logger.warn(
                    f" Warning: the input file seems not a raw count matrix. The prediction result may be biased")
            if (self.adata.n_vars >= 80000) or (len(self.adata.var_names[0]) >= 30) or (len(
                    self.adata.obs_names.intersection(pd.Index(['GAPDH', 'ACTB', 'CALM1', 'PTPRC']))) >= 1):
                raise ValueError(
                    f" The input matrix is detected to be a gene-by-cell matrix. Please provide a cell-by-gene matrix or add the input transpose option"
                )
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)
            self.indata = self.adata.X
            self.indata_genes = self.adata.var_names
            self.indata_names = self.adata.obs_names
        elif isinstance(filename, AnnData) or (isinstance(filename, str) and filename.endswith('.h5ad')):
            self.adata = sc.read(filename) if isinstance(filename, str) else filename
            self.adata.var_names_make_unique()
            if self.adata.X.min() < 0:
                logger.info(" Detected scaled expression in the input data, will try the `.raw` attribute")
                try:
                    self.indata = self.adata.raw.X
                    self.indata_genes = self.adata.raw.var_names
                    self.indata_names = self.adata.raw.obs_names
                except Exception as e:
                    raise Exception(f" Fail to use the `.raw` attribute in the input object. {e}")
            else:
                self.indata = self.adata.X
                self.indata_genes = self.adata.var_names
                self.indata_names = self.adata.obs_names
            if check_expression and np.abs(np.expm1(self.indata[0]).sum() - 10000) > 1:
                raise ValueError(
                    " Invalid expression matrix, expect log1p normalized expression to 10000 counts per cell")
        else:
            raise ValueError(
                " Invalid input. Supported types: .csv, .txt, .tsv, .tab, .mtx, .mtx.gz and .h5ad, or AnnData loaded in memory"
            )

        logger.info(f" Input data has {self.indata.shape[0]} cells and {len(self.indata_genes)} genes")

    def celltype(self, mode: str = 'best match', p_thres: float = 0.5) -> AnnotationResult:
        """Run celltyping jobs to predict cell types of input data.

        Parameters
        ----------
        mode: str optional
            The way cell prediction is performed.
            For each query cell, the default (`'best match'`) is to choose the cell type with the largest score/probability as the final prediction.
            Setting to `'prob match'` will enable a multi-label classification, which assigns 0 (i.e., unassigned), 1, or >=2 cell type labels to each query cell.
            (Default: `'best match'`)
        p_thres: float optional
            Probability threshold for the multi-label classification. Ignored if `mode` is `'best match'`.
            (Default: 0.5)

        Returns
        ----------
        :class:`~celltypist.classifier.AnnotationResult`
            An :class:`~celltypist.classifier.AnnotationResult` object. Four important attributes within this class are:
            1) :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`, predicted labels from celltypist.
            2) :attr:`~celltypist.classifier.AnnotationResult.decision_matrix`, decision matrix from celltypist.
            3) :attr:`~celltypist.classifier.AnnotationResult.probability_matrix`, probability matrix from celltypist.
            4) :attr:`~celltypist.classifier.AnnotationResult.adata`, AnnData object representation of the input data.

        """
        logger.info(f" Matching reference genes in the model")
        k_x = np.isin(self.indata_genes, self.model.classifier.features)
        if k_x.sum() == 0:
            raise ValueError(f" No features overlap with the model. Please provide gene symbols")
        else:
            logger.info(f" {k_x.sum()} features used for prediction")
        k_x_idx = np.where(k_x)[0]
        #self.indata = self.indata[:, k_x_idx]
        self.indata_genes = self.indata_genes[k_x_idx]
        lr_idx = pd.DataFrame(self.model.classifier.features,
                              columns=['features']).reset_index().set_index('features').loc[self.indata_genes,
                                                                                            'index'].values

        logger.info(f" Scaling input data")
        means_ = self.model.scaler.mean_[lr_idx]
        sds_ = self.model.scaler.scale_[lr_idx]
        self.indata = (self.indata[:, k_x_idx] - means_) / sds_
        self.indata[self.indata > 10] = 10

        ni, fs, cf = self.model.classifier.n_features_in_, self.model.classifier.features, self.model.classifier.coef_
        self.model.classifier.n_features_in_ = lr_idx.size
        self.model.classifier.features = self.model.classifier.features[lr_idx]
        self.model.classifier.coef_ = self.model.classifier.coef_[:, lr_idx]

        logger.info(" Predicting labels")
        decision_mat, prob_mat, lab = self.model.predict_labels_and_prob(self.indata, mode=mode, p_thres=p_thres)
        logger.info(" Prediction done!")

        #restore model after prediction
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
            Default to 5, 10, 15, 20, 25 and 30 for datasets with cell numbers less than 5k, 20k, 40k, 100k, 200k and above, respectively.

        Returns
        ----------
        :class:`~pandas.Series`
            A :class:`~pandas.Series` object showing the over-clustering result.

        """
        if 'connectivities' not in self.adata.obsp:
            logger.info(" Can not detect a neighborhood graph, will construct one before the over-clustering")
            adata = self.adata.copy()
            self.adata.obsm['X_pca'], self.adata.obsp['connectivities'], self.adata.obsp['distances'], self.adata.uns[
                'neighbors'] = Classifier._construct_neighbor_graph(adata)
        else:
            logger.info(
                " Detected a neighborhood graph in the input object, will run over-clustering on the basis of it")
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
        logger.info(f" Over-clustering input data with resolution set to {resolution}")
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
            An :class:`~celltypist.classifier.AnnotationResult` object containing the :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`.
        over_clustering: Union[list, tuple, np.ndarray, pd.Series, pd.Index]
            A list, tuple, numpy array, pandas series or index containing the over-clustering information.
        min_prop: float
            For the dominant cell type within a subcluster, the minimum proportion of cells required to support naming of the subcluster by this cell type.
            (Default: 0)

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


class Celltypist():
    r"""Build the ACTINN model.

    Parameters
    ----------
    classifier : Classification function
        Class that wraps the celltyping and majority voting processes, as defined above
    scaler : StandardScaler
        The scale factor for normalization.
    description : str
        text description of the model.

    """

    def __init__(self, clf=None, scaler=None, description=None):
        self.classifier = clf
        self.scaler = scaler
        self.description = description

    def fit(
            self,
            X=None,
            labels: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
            genes: Optional[Union[str, list, tuple, np.ndarray, pd.Series, pd.Index]] = None,
            transpose_input: bool = False,
            check_expression: bool = True,
            #LR param
            C: float = 1.0,
            solver: Optional[str] = None,
            max_iter: int = 1000,
            n_jobs: Optional[int] = None,
            #SGD param
            use_SGD: bool = False,
            alpha: float = 0.0001,
            #mini-batch
            mini_batch: bool = False,
            batch_number: int = 100,
            batch_size: int = 1000,
            epochs: int = 10,
            balance_cell_type: bool = False,
            #feature selection
            feature_selection: bool = False,
            top_genes: int = 300,
            #description
            date: str = '',
            details: str = '',
            url: str = '',
            source: str = '',
            version: str = '',
            #other param
            **kwargs):
        """Train a celltypist model using mini-batch (optional) logistic classifier with
        a global solver or stochastic gradient descent (SGD) learning.

        Parameters
        ----------
        X: Path optional
            Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData (h5ad).
            Also accepts the input as an :class:`~anndata.AnnData` object, or any array-like objects already loaded in memory.
            See `check_expression` for detailed format requirements.
            A cell-by-gene format is desirable (see `transpose_input` for more information).
        labels: Union[str, list, tuple, np.ndarray, pd.Series, pd.Index] optional
            Path to the file containing cell type label per line corresponding to the cells in `X`.
            Also accepts any list-like objects already loaded in memory (such as an array).
            If `X` is specified as an AnnData, this argument can also be set as a column name from cell metadata.
        genes: Union[str, list, tuple, np.ndarray, pd.Series, pd.Index] Optional
            Path to the file containing one gene per line corresponding to the genes in `X`.
            Also accepts any list-like objects already loaded in memory (such as an array).
            Note `genes` will be extracted from `X` where possible (e.g., `X` is an AnnData or data frame).
        transpose_input: bool
            Whether to transpose the input matrix. Set to `True` if `X` is provided in a gene-by-cell format.
            (Default: `False`)
        check_expression: bool optional
            Check whether the expression matrix in the input data is supplied as required.
            Except the case where a path to the raw count table file is specified, all other inputs for `X` should be in log1p normalized expression to 10000 counts per cell.
            Set to `False` if you want to train the data regardless of the expression formats.
            (Default: `True`)
        C: float optional
            Inverse of L2 regularization strength for traditional logistic classifier. A smaller value can possibly improve model generalization while at the cost of decreased accuracy.
            This argument is ignored if SGD learning is enabled (`use_SGD = True`).
            (Default: 1.0)
        solver: str optional
            Algorithm to use in the optimization problem for traditional logistic classifier.
            The default behavior is to choose the solver according to the size of the input data.
            This argument is ignored if SGD learning is enabled (`use_SGD = True`).
        max_iter: int optional
            Maximum number of iterations before reaching the minimum of the cost function.
            Try to decrease `max_iter` if the cost function does not converge for a long time.
            This argument is for both traditional and SGD logistic classifiers, and will be ignored if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
            (Default: 1000)
        n_jobs: int optional
            Number of CPUs used. Default to one CPU. `-1` means all CPUs are used.
            This argument is for both traditional and SGD logistic classifiers.
        use_SGD: bool optional
            Whether to implement SGD learning for the logistic classifier.
            (Default: `False`)
        alpha: float optional
            L2 regularization strength for SGD logistic classifier. A larger value can possibly improve model generalization while at the cost of decreased accuracy.
            This argument is ignored if SGD learning is disabled (`use_SGD = False`).
            (Default: 0.0001)
        mini_batch: bool optional
            Whether to implement mini-batch training for the SGD logistic classifier.
            Setting to `True` may improve the training efficiency for large datasets (for example, >100k cells).
            This argument is ignored if SGD learning is disabled (`use_SGD = False`).
            (Default: `False`)
        batch_number: int optional
            The number of batches used for training in each epoch. Each batch contains `batch_size` cells.
            For datasets which cannot be binned into `batch_number` batches, all batches will be used.
            This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
            (Default: 100)
        batch_size: int optional
            The number of cells within each batch.
            This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
            (Default: 1000)
        epochs: int optional
            The number of epochs for the mini-batch training procedure.
            The default values of `batch_number`, `batch_size`, and `epochs` together allow observing ~10^6 training cells.
            This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
            (Default: 10)
        balance_cell_type: bool optional
            Whether to balance the cell type frequencies in mini-batches during each epoch.
            Setting to `True` will sample rare cell types with a higher probability, ensuring close-to-even cell type distributions in mini-batches.
            This argument is relevant only if mini-batch SGD training is conducted (`use_SGD = True` and `mini_batch = True`).
            (Default: `False`)
        feature_selection: bool optional
            Whether to perform two-pass data training where the first round is used for selecting important features/genes using SGD learning.
            If `True`, the training time will be longer.
            (Default: `False`)
        top_genes: int optional
            The number of top genes selected from each class/cell-type based on their absolute regression coefficients.
            The final feature set is combined across all classes (i.e., union).
            (Default: 300)
        date: str optional
            Free text of the date of the model. Default to the time when the training is completed.
        details: str optional
            Free text of the description of the model.
        url: str optional
            Free text of the (possible) download url of the model.
        source: str optional
            Free text of the source (publication, database, etc.) of the model.
        version: str optional
            Free text of the version of the model.
        **kwargs
            Other keyword arguments passed to :class:`~sklearn.linear_model.LogisticRegression` (`use_SGD = False`) or :class:`~sklearn.linear_model.SGDClassifier` (`use_SGD = True`).

        Returns
        -------
        :class:`~celltypist.models.Model`
            An instance of the :class:`~celltypist.models.Model` trained by celltypist.

        """
        #prepare
        logger.info(" Preparing data before training")
        indata, labels, genes = prepare_data_celltypist(X, labels, genes, transpose_input)
        indata = to_array_celltypist(indata)
        labels = np.array(labels)
        genes = np.array(genes)
        #check
        if check_expression and (np.abs(np.expm1(indata[0]).sum() - 10000) > 1):
            raise ValueError(" Invalid expression matrix, expect log1p normalized expression to 10000 counts per cell")
        if len(labels) != indata.shape[0]:
            raise ValueError(
                f" Length of training labels ({len(labels)}) does not match the number of input cells ({indata.shape[0]})"
            )
        if len(genes) != indata.shape[1]:
            raise ValueError(
                f" The number of genes ({len(genes)}) provided does not match the number of genes in the training data ({indata.shape[1]})"
            )
        #filter
        flag = indata.sum(axis=0) == 0
        if flag.sum() > 0:
            logger.info(f" {flag.sum()} non-expressed genes are filtered out")
            #indata = indata[:, ~flag]
            genes = genes[~flag]
        #scaler
        logger.info(f" Scaling input data")
        scaler = StandardScaler()
        indata = scaler.fit_transform(indata[:, ~flag] if flag.sum() > 0 else indata)
        indata[indata > 10] = 10
        #classifier
        if use_SGD or feature_selection:
            classifier = SGDClassifier_celltypist(indata=indata, labels=labels, alpha=alpha, max_iter=max_iter,
                                                  n_jobs=n_jobs, mini_batch=mini_batch, batch_number=batch_number,
                                                  batch_size=batch_size, epochs=epochs,
                                                  balance_cell_type=balance_cell_type, **kwargs)
        else:
            classifier = LRClassifier_celltypist(indata=indata, labels=labels, C=C, solver=solver, max_iter=max_iter,
                                                 n_jobs=n_jobs, **kwargs)
        #feature selection -> new classifier and scaler
        if feature_selection:
            logger.info(f" Selecting features")
            if len(genes) <= top_genes:
                raise ValueError(
                    f" The number of genes ({len(genes)}) is fewer than the `top_genes` ({top_genes}). Unable to perform feature selection"
                )
            gene_index = np.argpartition(np.abs(classifier.coef_), -top_genes, axis=1)[:, -top_genes:]
            gene_index = np.unique(gene_index)
            logger.info(f" {len(gene_index)} features are selected")
            genes = genes[gene_index]
            #indata = indata[:, gene_index]
            logger.info(f" Starting the second round of training")
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
        #model finalization
        classifier.features = genes
        if not date:
            date = str(datetime.now())
        description = {
            'date': date,
            'details': details,
            'url': url,
            'source': source,
            'version': version,
            'number_celltypes': len(classifier.classes_)
        }
        logger.info(f" Model training done!")

        self.classifier = classifier
        self.scaler = scaler
        self.description = description

    def predict(self, filename: Union[AnnData, str] = "", check_expression: bool = False, load_model: bool = False,
                model: Optional[Union[str, Model]] = None, transpose_input: bool = False,
                gene_file: Optional[str] = None, cell_file: Optional[str] = None, mode: str = 'best match',
                p_thres: float = 0.5, majority_voting: bool = False,
                over_clustering: Optional[Union[str, list, tuple, np.ndarray, pd.Series,
                                                pd.Index]] = None, min_prop: float = 0) -> AnnotationResult:
        """Run the prediction and (optional) majority voting to annotate the input
        dataset.

        Parameters
        ----------
        filename: Union[AnnData,str]  optional
            Path to the input count matrix (supported types are csv, txt, tsv, tab and mtx) or AnnData (h5ad).
            If it's the former, a cell-by-gene format is desirable (see `transpose_input` for more information).
            Also accepts the input as an :class:`~anndata.AnnData` object already loaded in memory.
            Genes should be gene symbols. Non-expressed genes are preferred to be provided as well.
        model: Union[str, Model] optional
            Model used to predict the input cells. Default to using the `'Immune_All_Low.pkl'` model.
            Can be a :class:`~celltypist.models.Model` object that wraps the logistic Classifier and the StandardScaler, the
            path to the desired model file, or the model name.
            To see all available models and their descriptions, use :func:`~celltypist.models.models_description`.
        transpose_input: boolUnion[str, Model]
            Whether to transpose the input matrix. Set to `True` if `filename` is provided in a gene-by-cell format.
            (Default: `False`)
        gene_file: str optional
            Path to the file which stores each gene per line corresponding to the genes used in the provided mtx file.
            Ignored if `filename` is not provided in the mtx format.
        cell_file: str optional
            Path to the file which stores each cell per line corresponding to the cells used in the provided mtx file.
            Ignored if `filename` is not provided in the mtx format.
        mode: str optional
            The way cell prediction is performed.
            For each query cell, the default (`'best match'`) is to choose the cell type with the largest score/probability as the final prediction.
            Setting to `'prob match'` will enable a multi-label classification, which assigns 0 (i.e., unassigned), 1, or >=2 cell type labels to each query cell.
            (Default: `'best match'`)
        p_thres: float optional
            Probability threshold for the multi-label classification. Ignored if `mode` is `'best match'`.
            (Default: 0.5)
        majority_voting: bool optional
            Whether to refine the predicted labels by running the majority voting classifier after over-clustering.
            (Default: `False`)
        over_clustering: Union[str, list, tuple, np.ndarray, pd.Series, pd.Index] optional
            This argument can be provided in several ways:
            1) an input plain file with the over-clustering result of one cell per line.
            2) a string key specifying an existing metadata column in the AnnData (pre-created by the user).
            3) a python list, tuple, numpy array, pandas series or index representing the over-clustering result of the input cells.
            4) if none of the above is provided, will use a heuristic over-clustering approach according to the size of input data.
            Ignored if `majority_voting` is set to `False`.
        min_prop: float optional
            For the dominant cell type within a subcluster, the minimum proportion of cells required to support naming of the subcluster by this cell type.
            Ignored if `majority_voting` is set to `False`.
            Subcluster that fails to pass this proportion threshold will be assigned `'Heterogeneous'`.
            (Default: 0)

        Returns
        ----------
        output :class:`~celltypist.classifier.AnnotationResult`
            An :class:`~celltypist.classifier.AnnotationResult` object. Four important attributes within this class are:
            1) :attr:`~celltypist.classifier.AnnotationResult.predicted_labels`, predicted labels from celltypist.
            2) :attr:`~celltypist.classifier.AnnotationResult.decision_matrix`, decision matrix from celltypist.
            3) :attr:`~celltypist.classifier.AnnotationResult.probability_matrix`, probability matrix from celltypist.
            4) :attr:`~celltypist.classifier.AnnotationResult.adata`, AnnData representation of the input data.

        """
        #load model
        # lr_classifier = Model(self.classifier, self.scaler, self.description) if isinstance(model, Model) else Model.load(model)
        if load_model:
            lr_classifier = Model.load(model)
        else:
            lr_classifier = Model(self.classifier, self.scaler, self.description)
        #construct Classifier class
        clf = Classifier(filename=filename, model=lr_classifier, transpose=transpose_input, gene_file=gene_file,
                         cell_file=cell_file, check_expression=check_expression)
        #predict
        predictions = clf.celltype(mode=mode, p_thres=p_thres)
        if not majority_voting:
            return predictions
        if predictions.cell_count <= 50:
            logger.warn(
                f" Warning: the input number of cells ({predictions.cell_count}) is too few to conduct proper over-clustering; no majority voting is performed"
            )
            return predictions
        #over clustering
        if over_clustering is None:
            over_clustering = clf.over_cluster()
            predictions.adata = clf.adata
        elif isinstance(over_clustering, str):
            if over_clustering in clf.adata.obs:
                over_clustering = clf.adata.obs[over_clustering]
            else:
                logger.info(
                    f" Did not identify '{over_clustering}' as a cell metadata column, assume it to be a plain text file"
                )
                try:
                    with open(over_clustering) as f:
                        over_clustering = [x.strip() for x in f.readlines()]
                except Exception as e:
                    raise Exception(f" {e}")
        if len(over_clustering) != clf.adata.n_obs:
            raise ValueError(
                f" Length of `over_clustering` ({len(over_clustering)}) does not match the number of input cells ({clf.adata.n_obs})"
            )
        #majority voting
        print(predictions)
        return Classifier.majority_vote(predictions, over_clustering, min_prop=min_prop)

    def score(self, input_adata, predictions, labels, map, label_conversion=False):
        """Run the prediction and (optional) majority voting to evaluate the model
        performance.

        Parameters
        ----------
        input_adata: Anndata
            Input data anndata with label ground truth
        predictions: classifier.AnnotationResult
            Output from prediction function.
        labels : string
            column name for annotated cell types in the input Anndata
        label_conversion: boolean optional
            whether to match predicted labels to annotated cell type labels provided in input_adata
        map: dictionary
            a dictionary for label conversion

        Returns
        -------
        correct: int
            Number of correct predictions
        Accuracy: float
            Prediction accuracy from the model

        """
        pred_labels = np.array(predictions.predicted_labels)[:, 0]
        if label_conversion:
            correct = 0
            #for i, cell in enumerate(np.array(adVal.obs.Cell_type)):
            for i, cell in enumerate(np.array(input_adata.obs[labels])):
                if pred_labels[i] in map[cell]:
                    correct += 1
        else:
            correct = sum(np.array(input_adata.obs[labels]) == pred_labels)
        return (correct / len(predictions.predicted_labels))
