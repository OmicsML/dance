import collections
import os
import os.path as osp
import pprint
import shutil
import sys
from glob import glob

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from dance import logger
from dance.config import METADIR
from dance.data import Data
from dance.datasets.base import BaseDataset
from dance.metadata.imputation import IMPUTATION_DATASET_TO_FILE
from dance.registers import register_dataset
from dance.typing import Dict, List, Optional, Set, Tuple
from dance.utils.download import download_file, download_unzip
from dance.utils.io import load_data_url_dict_from_csv
from dance.utils.preprocess import cell_label_to_df


def _load_scdeepsort_metadata():
    path = METADIR / "scdeepsort.csv"
    logger.debug(f"Loading scdeepsort metadata from {path}")
    scdeepsort_meta_df = pd.read_csv(path).astype(str)

    bench_url_dict, available_data = {}, []
    for _, i in scdeepsort_meta_df.iterrows():
        bench_url_dict[i["celltype_fname"]] = i["celltype_url"]
        bench_url_dict[i["data_fname"]] = i["data_url"]
        available_data.append({key: i[key] for key in ("split", "species", "tissue", "dataset")})

    return bench_url_dict, available_data


@register_dataset("CellTypeAnnotation")
class CellTypeAnnotationDataset(BaseDataset):
    _DISPLAY_ATTRS = ("species", "tissue", "train_dataset", "test_dataset")
    ALL_URL_DICT: Dict[str, str] = {
        "train_human_cell_atlas": "https://www.dropbox.com/s/1itq1pokplbqxhx?dl=1",
        "test_human_test_data": "https://www.dropbox.com/s/gpxjnnvwyblv3xb?dl=1",
        "train_mouse_cell_atlas": "https://www.dropbox.com/s/ng8d3eujfah9ppl?dl=1",
        "test_mouse_test_data": "https://www.dropbox.com/s/pkr28czk5g3al2p?dl=1",
    }  # yapf: disable
    BENCH_URL_DICT, AVAILABLE_DATA = _load_scdeepsort_metadata()

    def __init__(self, full_download=False, train_dataset=None, test_dataset=None, species=None, tissue=None,
                 train_dir="train", test_dir="test", map_path="map", data_dir="./"):
        super().__init__(data_dir, full_download)

        self.data_dir = data_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.species = species
        self.tissue = tissue
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.map_path = map_path

    def download_all(self):
        if self.is_complete():
            return

        # Download and overwrite
        for name, url in self.ALL_URL_DICT.items():
            download_unzip(url, self.data_dir)

            parts = name.split("_")  # [train|test]_{species}_[cell|test]_atlas
            download_path = osp.join(self.data_dir, "_".join(parts[1:]))
            move_path = osp.join(self.data_dir, *parts[:2])

            os.makedirs(osp.dirname(move_path), exist_ok=True)
            try:
                shutil.rmtree(move_path)
            except FileNotFoundError:
                pass
            os.rename(download_path, move_path)

    def get_all_filenames(self, filetype: str = "csv", feat_suffix: str = "data", label_suffix: str = "celltype"):
        filenames = []
        for id in self.train_dataset + self.test_dataset:
            filenames.append(f"{self.species}_{self.tissue}{id}_{feat_suffix}.{filetype}")
            filenames.append(f"{self.species}_{self.tissue}{id}_{label_suffix}.{filetype}")
        return filenames

    def download(self, download_map=True):
        if self.is_complete():
            return

        filenames = self.get_all_filenames()
        # Download training and testing data
        for name, url in self.BENCH_URL_DICT.items():
            parts = name.split("_")  # [train|test]_{species}_{tissue}{id}_[celltype|data].csv
            filename = "_".join(parts[1:])
            if filename in filenames:
                filepath = osp.join(self.data_dir, *parts[:2], filename)
                download_file(url, filepath)

        if download_map:
            # Download mapping data
            download_unzip("https://www.dropbox.com/sh/hw1189sgm0kfrts/AAAapYOblLApqygZ-lGo_70-a?dl=1",
                           osp.join(self.data_dir, "map"))

    def is_complete_all(self):
        """Check if data is complete."""
        check = [
            osp.join(self.data_dir, "train"),
            osp.join(self.data_dir, "test"),
            osp.join(self.data_dir, "pretrained")
        ]
        for i in check:
            if not osp.exists(i):
                logger.info(f"file {i} doesn't exist")
                return False
        return True

    def is_complete(self):
        """Check if benchmarking data is complete."""
        for name in self.BENCH_URL_DICT:
            if any(i not in name for i in (self.species, self.tissue)):
                continue
            filename = name[name.find(self.species):]
            file_i = osp.join(self.data_dir, *(name.split("_"))[:2], filename)
            if not osp.exists(file_i):
                logger.info(file_i)
                logger.info(f"file {filename} doesn't exist")
                return False
        # check maps
        map_check = [
            osp.join(self.data_dir, "map", "mouse", "map.xlsx"),
            osp.join(self.data_dir, "map", "human", "map.xlsx"),
            osp.join(self.data_dir, "map", "celltype2subtype.xlsx")
        ]
        for file in map_check:
            if not osp.exists(file):
                logger.info(f"file {name} doesn't exist")
                return False
        return True

    def _load_raw_data(self, ct_col: str = "Cell_type") -> Tuple[ad.AnnData, List[Set[str]], List[str], int]:
        species = self.species
        tissue = self.tissue
        train_dataset_ids = self.train_dataset
        test_dataset_ids = self.test_dataset
        data_dir = self.data_dir
        train_dir = osp.join(data_dir, self.train_dir)
        test_dir = osp.join(data_dir, self.test_dir)
        map_path = osp.join(data_dir, self.map_path, self.species)

        # Load raw data
        train_feat_paths, train_label_paths = self._get_data_paths(train_dir, species, tissue, train_dataset_ids)
        test_feat_paths, test_label_paths = self._get_data_paths(test_dir, species, tissue, test_dataset_ids)
        train_feat, test_feat = (self._load_dfs(paths, transpose=True) for paths in (train_feat_paths, test_feat_paths))
        train_label, test_label = (self._load_dfs(paths) for paths in (train_label_paths, test_label_paths))

        # Combine features (only use features that are present in the training data)
        train_size = train_feat.shape[0]
        feat_df = pd.concat(train_feat.align(test_feat, axis=1, join="left", fill_value=0)).fillna(0)
        adata = ad.AnnData(feat_df, dtype=np.float32)

        # Convert cell type labels and map test cell type names to train
        cell_types = set(train_label[ct_col].unique())
        idx_to_label = sorted(cell_types)
        cell_type_mappings: Dict[str, Set[str]] = self.get_map_dict(map_path, tissue)
        train_labels, test_labels = train_label[ct_col].tolist(), []
        for i in test_label[ct_col]:
            test_labels.append(i if i in cell_types else cell_type_mappings.get(i))
        labels: List[Set[str]] = train_labels + test_labels

        logger.debug("Mapped test cell-types:")
        for i, j, k in zip(test_label.index, test_label[ct_col], test_labels):
            logger.debug(f"{i}:{j}\t-> {k}")

        logger.info(f"Loaded expression data: {adata}")
        logger.info(f"Number of training samples: {train_feat.shape[0]:,}")
        logger.info(f"Number of testing samples: {test_feat.shape[0]:,}")
        logger.info(f"Cell-types (n={len(idx_to_label)}):\n{pprint.pformat(idx_to_label)}")

        return adata, labels, idx_to_label, train_size

    def _raw_to_dance(self, raw_data):
        adata, cell_labels, idx_to_label, train_size = raw_data
        adata.obsm["cell_type"] = cell_label_to_df(cell_labels, idx_to_label, index=adata.obs.index)
        data = Data(adata, train_size=train_size)
        return data

    @staticmethod
    def _get_data_paths(data_dir: str, species: str, tissue: str, dataset_ids: List[str], *, filetype: str = "csv",
                        feat_suffix: str = "data", label_suffix: str = "celltype") -> Tuple[List[str], List[str]]:
        feat_paths, label_paths = [], []
        for path_list, suffix in zip((feat_paths, label_paths), (feat_suffix, label_suffix)):
            for i in dataset_ids:
                path_list.append(osp.join(data_dir, species, f"{species}_{tissue}{i}_{suffix}.{filetype}"))
        return feat_paths, label_paths

    @staticmethod
    def _load_dfs(paths: List[str], *, index_col: Optional[int] = 0, transpose: bool = False, **kwargs):
        dfs = []
        for path in paths:
            logger.info(f"Loading data from {path}")
            # TODO: load feat as csr
            df = pd.read_csv(path, index_col=index_col, **kwargs)
            # Labels: cell x cell-type; Data: feature x cell (need to transpose)
            df = df.T if transpose else df
            # Add dataset info to index
            dataset_name = "_".join(osp.basename(path).split("_")[:-1])
            df.index = dataset_name + "_" + df.index.astype(str)
            dfs.append(df)
        combined_df = pd.concat(dfs)
        return combined_df

    @staticmethod
    def get_map_dict(map_file_path: str, tissue: str) -> Dict[str, Set[str]]:
        """Load cell-type mappings.

        Parameters
        ----------
        map_file_path
            Path to the mapping file.
        tissue
            Tissue of interest.

        Notes
        -----
        Merge mapping across all test sets for the required tissue.

        """
        map_df = pd.read_excel(osp.join(map_file_path, "map.xlsx"))
        map_dict = collections.defaultdict(set)
        for _, row in map_df.iterrows():
            if row["Tissue"] == tissue:
                map_dict[row["Celltype"]].add(row["Training dataset cell type"])
        return dict(map_dict)


@register_dataset("clustering")
class ClusteringDataset(BaseDataset):
    """Data downloading and loading for clustering.

    Parameters
    ----------
    data_dir
        Path to store datasets.
    dataset
        Choice of dataset. Available options are '10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell'.

    """

    URL_DICT = load_data_url_dict_from_csv(METADIR / "clustering.csv")
    AVAILABLE_DATA = sorted(URL_DICT)

    def __init__(self, data_dir: str = "./data", dataset: str = "mouse_bladder_cell"):
        super().__init__(data_dir, full_download=False)
        self.data_dir = data_dir
        self.dataset = dataset

    @property
    def data_path(self) -> str:
        return osp.join(self.data_dir, f"{self.dataset}.h5")

    def download(self):
        download_file(self.URL_DICT[self.dataset], self.data_path)

    def is_complete(self):
        return osp.exists(self.data_path)

    def _load_raw_data(self) -> Tuple[ad.AnnData, np.ndarray]:
        with h5py.File(self.data_path, "r") as f:
            x = np.array(f["X"])
            y = np.array(f["Y"])
        adata = ad.AnnData(x, dtype=np.float32)
        return adata, y

    def _raw_to_dance(self, raw_data: Tuple[ad.AnnData, np.ndarray]):
        adata, y = raw_data
        adata.obsm["Group"] = y
        data = Data(adata, train_size="all")
        return data


@register_dataset("imputation")
class ImputationDataset(BaseDataset):
    URL = load_data_url_dict_from_csv(METADIR / "imputation.csv")
    DATASET_TO_FILE =IMPUTATION_DATASET_TO_FILE   # yapf: disable
    AVAILABLE_DATA = sorted(URL)

    def __init__(self, data_dir="data", dataset="human_stemcell", train_size=0.1):
        super().__init__(data_dir, full_download=False)
        self.data_dir = data_dir
        self.dataset = dataset
        self.train_size = train_size

    def download(self):

        gene_class = [
            "pbmc_data", "mouse_brain_data", "mouse_embryo_data", "human_stemcell_data", "human_breast_TGFb_data",
            "human_breast_Dox_data", "human_melanoma_data", "mouse_visual_data"
        ]
        assert self.dataset in gene_class
        class_name = self.dataset

        file_name = {
            "pbmc_data": "5k.zip?dl=0",
            "mouse_embryo_data": "GSE65525.zip?dl=0",
            "mouse_brain_data": "neuron_10k.zip?dl=0",
            "human_stemcell_data": "GSE75748.zip?dl=0",
            "human_breast_TGFb_data": "GSE114397.zip?dl=0",
            "human_breast_Dox_data": "GSM3141014.zip?dl=0",
            "human_melanoma_data": "human_melanoma_data.zip?dl=0",
            "mouse_visual_data": "mouse_visual_data.zip?dl=0"
        }

        dl_files = {
            "pbmc_data": "5k_*",
            "mouse_embryo_data": "GSE65525",
            "mouse_brain_data": "neuron*",
            "human_stemcell_data": "GSE75748",
            "human_breast_TGFb_data": "GSE11*",
            "human_breast_Dox_data": "GSM31*",
            "human_melanoma_data": "human*",
            "mouse_visual_data": "GSM27*"
        }

        if sys.platform != 'win32':
            if not osp.exists(self.data_dir):
                os.system("mkdir " + self.data_dir)
            if not osp.exists(self.data_dir + "/train"):
                os.system("mkdir " + self.data_dir + "/train")

            if not any(map(osp.exists, glob(osp.join(self.data_dir, "train", class_name, dl_files[class_name])))):
                os.system("mkdir " + self.data_dir + "/train/" + class_name)
                os.system("wget " + self.URL[class_name])  # assumes linux... mac needs to install
                os.system("unzip " + file_name[class_name])
                os.system("rm " + file_name[class_name])
                os.system("mv " + dl_files[class_name] + " " + self.data_dir + "/train/" + class_name + "/")
            os.system("cp -r " + self.data_dir + "/train/ " + self.data_dir + "/test")
        else:
            if not osp.exists(self.data_dir):
                os.system("mkdir " + self.data_dir)
            if not osp.exists(self.data_dir + "/train"):
                os.mkdir(self.data_dir + "/train")

            if not any(map(osp.exists, glob(osp.join(self.data_dir, "train", class_name, dl_files[class_name])))):
                os.mkdir(self.data_dir + "/train/" + class_name)
                os.system("curl " + self.URL[class_name])
                os.system("tar -xf " + file_name[class_name])
                os.system("del -R " + file_name[class_name])
                os.system("move " + dl_files[class_name] + " " + self.data_dir + "/train/" + class_name + "/")
            os.system("copy /r " + self.data_dir + "/train/ " + self.data_dir + "/test")

    def is_complete(self):
        # check whether data is complete or not
        check = [osp.join(self.data_dir, i, self.dataset) for i in ("train", "test")]

        for i in check:
            if not osp.exists(i):
                logger.info("file {} doesn't exist".format(i))
                return False
        return True

    def _load_raw_data(self) -> ad.AnnData:
        if self.dataset[-5:] != '_data':
            dataset = self.dataset + '_data'
        else:
            dataset = self.dataset

        if self.dataset == 'mouse_embryo' or self.dataset == 'mouse_embryo_data' or self.dataset == "mouse_visual_data":
            for i in range(len(self.DATASET_TO_FILE[dataset])):
                fname = self.DATASET_TO_FILE[dataset][i]
                data_path = f'{self.data_dir}/train/{dataset}/{fname}'
                if i == 0:
                    counts = pd.read_csv(data_path, header=None, index_col=0)
                    time = pd.Series(np.zeros(counts.shape[1]))
                else:
                    x = pd.read_csv(data_path, header=None, index_col=0)
                    time = pd.concat([time, pd.Series(np.zeros(x.shape[1])) + i])
                    counts = pd.concat([counts, x], axis=1)
            time = pd.DataFrame(time)
            time.columns = ['time']
            counts = counts.T
            counts.index = [i for i in range(counts.shape[0])]
            adata = ad.AnnData(csr_matrix(counts.values))
            adata.var_names = counts.columns.tolist()
            adata.obs['time'] = time.to_numpy()
        else:
            data_path = osp.join(self.data_dir, "train", dataset, self.DATASET_TO_FILE[dataset])
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"{data_path} does not exist")

            if self.DATASET_TO_FILE[dataset][-3:] == 'csv':
                counts = pd.read_csv(data_path, header=None, index_col=0)
                nums = pd.Series(np.arange(counts.shape[1]))
                nums = pd.DataFrame(nums)
                nums.columns = ['nums']
                counts = counts.T
                counts.index = [i for i in range(counts.shape[0])]
                adata = ad.AnnData(csr_matrix(counts.values))
                adata.var_names = counts.columns.tolist()
                adata.obs['nums'] = nums.to_numpy()
            if self.DATASET_TO_FILE[dataset][-2:] == 'gz':
                counts = pd.read_csv(data_path, index_col=0, compression='gzip', header=0)
                counts = counts.T
                adata = ad.AnnData(csr_matrix(counts.values))
                # adata.obs_names = ["%d" % i for i in range(adata.shape[0])]
                adata.obs_names = counts.index.tolist()
                adata.var_names = counts.columns.tolist()
            elif self.DATASET_TO_FILE[dataset][-2:] == 'h5':
                adata = sc.read_10x_h5(data_path)
                adata.var_names_make_unique()

        return adata

    def _raw_to_dance(self, raw_data: ad.AnnData):
        adata = raw_data
        data = Data(adata, train_size=int(adata.n_obs * self.train_size))
        return data
