import collections
import glob
import os
import os.path as osp
import pprint
import sys
from dataclasses import dataclass

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dance import logger
from dance.data import download_file, download_unzip
from dance.transforms.preprocess import load_imputation_data_internal
from dance.typing import Dict, List, Optional, Set, Tuple


class CellTypeDataset:

    def __init__(self, download_all=False, train_dataset=None, test_dataset=None, species=None, tissue=None,
                 train_dir="train", test_dir="test", map_path="map", filetype="csv", threshold=None, exclude_rate=None,
                 data_dir="./"):
        self.data_dir = data_dir
        self.download_all = download_all
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.species = species
        self.tissue = tissue
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.map_path = map_path
        self.filetype = filetype
        self.threshold = threshold
        self.exclude_rate = exclude_rate

    def download_all_data(self):
        # download data
        gene_class = ["human_cell_atlas", "human_test_data", "mouse_cell_atlas", "mouse_test_data"]

        url = {
            "human_cell_atlas": "https://www.dropbox.com/s/1itq1pokplbqxhx/human_cell_atlas.zip?dl=0",
            "human_test_data": "https://www.dropbox.com/s/gpxjnnvwyblv3xb/human_test_data.zip?dl=0",
            "mouse_cell_atlas": "https://www.dropbox.com/s/ng8d3eujfah9ppl/mouse_cell_atlas.zip?dl=0",
            "mouse_test_data": "https://www.dropbox.com/s/pkr28czk5g3al2p/mouse_test_data.zip?dl=0"
        }

        file_name = {
            "human_cell_atlas": "human_cell_atlas.zip?dl=0",
            "human_test_data": "human_test_data.zip?dl=0",
            "mouse_cell_atlas": "mouse_cell_atlas.zip?dl=0",
            "mouse_test_data": "mouse_test_data.zip?dl=0"
        }

        species = {
            "human_cell_atlas": "human",
            "human_test_data": "human",
            "mouse_cell_atlas": "mouse",
            "mouse_test_data": "mouse"
        }

        os.system("mkdir " + self.data_dir + "/train")
        os.system("mkdir " + self.data_dir + "/train/mouse")
        os.system("mkdir " + self.data_dir + "/train/human")
        for class_name in gene_class:
            os.system("wget " + url[class_name])
            os.system("unzip " + file_name[class_name])
            os.system("mv " + class_name + "/* " + self.data_dir + "/train/" + species[class_name] + "/")
            os.system("rm " + file_name[class_name])
            os.system("rm -r " + class_name)

        os.system("cp -r " + self.data_dir + "/train/ " + self.data_dir + "/test")

    def download_benchmark_data(self, download_map=True, download_pretrained=True):

        if self.is_benchmark_complete():
            return

        urls = {
            # Mouse spleen benchmark
            "train_mouse_Spleen1970_celltype.csv":  "https://www.dropbox.com/s/3ea64vk546fjxvr?dl=1",
            "train_mouse_Spleen1970_data.csv":      "https://www.dropbox.com/s/c4te0fr1qicqki8?dl=1",
            "test_mouse_Spleen1759_celltype.csv":   "https://www.dropbox.com/s/gczehvgai873mhb?dl=1",
            "test_mouse_Spleen1759_data.csv":       "https://www.dropbox.com/s/fl8t7rbo5dmznvq?dl=1",
            # Mouse brain benchmark
            "train_mouse_Brain753_celltype.csv":    "https://www.dropbox.com/s/x2katwk93z06sgw?dl=1",
            "train_mouse_Brain753_data.csv":        "https://www.dropbox.com/s/3f3wbplgo3xa4ww?dl=1",
            "train_mouse_Brain3285_celltype.csv":   "https://www.dropbox.com/s/ozsobozk3ihkrqg?dl=1",
            "train_mouse_Brain3285_data.csv":       "https://www.dropbox.com/s/zjrloejx8iqdqsa?dl=1",
            "test_mouse_Brain2695_celltype.csv":    "https://www.dropbox.com/s/gh72dk7i0p7fggu?dl=1",
            "test_mouse_Brain2695_data.csv":        "https://www.dropbox.com/s/ufianih66xjqxdu?dl=1",
            # Mouse kidney benchmark
            "train_mouse_Kidney4682_celltype.csv":  "https://www.dropbox.com/s/3plrve7g9v428ec?dl=1",
            "train_mouse_Kidney4682_data.csv":      "https://www.dropbox.com/s/olf5nirtieu1ikq?dl=1",
            "test_mouse_Kidney203_celltype.csv":    "https://www.dropbox.com/s/t4eyaig889qdiz2?dl=1",
            "test_mouse_Kidney203_data.csv":        "https://www.dropbox.com/s/kmos1ceubumgmpj?dl=1",
        }  # yapf: disable

        # Download training and testing data
        for name, url in urls.items():
            parts = name.split("_")  # [train|test]_{species}_{tissue}{id}_[celltype|data].csv
            filename = "_".join(parts[1:])
            filepath = osp.join(self.data_dir, *parts[:2], filename)
            download_file(url, filepath)

        if download_map:
            # Download mapping data
            download_unzip("https://www.dropbox.com/sh/hw1189sgm0kfrts/AAAapYOblLApqygZ-lGo_70-a?dl=1",
                           osp.join(self.data_dir, "map"))

        if download_pretrained:
            # Download pretrained stats data
            download_unzip("https://www.dropbox.com/sh/s2cxcrzl2ama9zp/AACKwiYtS8hbOOudQLIMDvXUa?dl=1",
                           osp.join(self.data_dir, "pretrained"))

    def is_complete(self):
        """Check if data is complete."""
        check = [
            osp.join(self.data_dir, "train"),
            osp.join(self.data_dir, "test"),
            osp.join(self.data_dir, "pretrained")
        ]

        for i in check:
            if not osp.exists(i):
                print(f"file {i} doesn't exist")
                return False
        return True

    def is_benchmark_complete(self):
        check = [
            "test_mouse_Brain2695_celltype.csv",
            "test_mouse_Brain2695_data.csv",
            "test_mouse_Kidney203_celltype.csv",
            "test_mouse_Kidney203_data.csv",
            "test_mouse_Spleen1759_celltype.csv",
            "test_mouse_Spleen1759_data.csv",
            "train_mouse_Brain3285_celltype.csv",
            "train_mouse_Brain3285_data.csv",
            "train_mouse_Brain753_celltype.csv",
            "train_mouse_Brain753_data.csv",
            "train_mouse_Kidney4682_celltype.csv",
            "train_mouse_Kidney4682_data.csv",
            "train_mouse_Spleen1970_celltype.csv",
            "train_mouse_Spleen1970_data.csv",
        ]
        for name in check:
            filename = name[name.find('mouse'):]
            file_i = osp.join(self.data_dir, *name.split("_")[:2], filename)
            if not osp.exists(file_i):
                print(file_i)
                print(f"file {filename} doesn't exist")
                return False
        # check maps
        map_check = [
            osp.join(self.data_dir, "map", "mouse", "map.xlsx"),
            osp.join(self.data_dir, "map", "human", "map.xlsx"),
            osp.join(self.data_dir, "map", "celltype2subtype.xlsx")
        ]
        for file in map_check:
            if not osp.exists(file):
                print(f"file {name} doesn't exist")
                return False
        # TODO: check pretrained data
        return True

    def load_data(self):
        # Load data from existing files, or download files and load data.
        if self.download_all:
            self.download_all_data()
        elif not self.is_complete():
            self.download_benchmark_data()
        return self._load_data()

    def _load_data(self, ct_col: str = "Cell_type"):
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


class ClusteringDataset():
    """Data downloading and loading for clustering.

    Args:
            data_dir (str, optional): Path to store datasets. Defaults to "./data".
            dataset (str, optional): Choice of dataset. Defaults to "mouse_bladder_cell",
                                     choices=['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell'].

    """

    def __init__(self, data_dir="./data", dataset="mouse_bladder_cell"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.X = None
        self.Y = None

    def download_data(self):
        # download data
        data_url = {
            "10X_PBMC": "https://www.dropbox.com/s/pfunm27qzgfpj3u/10X_PBMC.h5?dl=1",
            "mouse_bladder_cell": "https://www.dropbox.com/s/xxtnomx5zrifdwi/mouse_bladder_cell.h5?dl=1",
            "mouse_ES_cell": "https://www.dropbox.com/s/zbuku7oznvji8jk/mouse_ES_cell.h5?dl=1",
            "worm_neuron_cell": "https://www.dropbox.com/s/58fkgemi2gcnp2k/worm_neuron_cell.h5?dl=1"
        }
        data_name = self.dataset
        download_file(data_url.get(data_name), self.data_dir + "/{}.h5".format(data_name))
        return self

    def is_complete(self):
        # judge data is complete or not
        return osp.exists(osp.join(self.data_dir, f"{self.dataset}.h5"))

    def load_data(self):
        # Load data from existing h5ad files, or download files and load data.
        if self.is_complete():
            pass
        else:
            self.download_data()
            assert self.is_complete()

        data_mat = h5py.File(f"{self.data_dir}/{self.dataset}.h5", "r")
        X = np.array(data_mat["X"])
        adata = ad.AnnData(X, dtype=np.float32)
        Y = np.array(data_mat["Y"])
        return adata, Y


class PretrainDataset(Dataset):
    """Dataset object for scDSC pretraining."""

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


class TrainingDataset(Dataset):
    """Dataset object for scDSC training."""

    def __init__(self, X, Y):
        self.x = X
        self.y = Y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))


@dataclass
class ImputationDatasetParams:
    data_dir = None
    random_seed = None
    min_counts = None
    train_dataset = None
    test_dataset = None
    gpu = None
    filetype = None


class ImputationDataset():

    def __init__(self, random_seed=10, gpu=-1, filetype=None, data_dir="data", train_dataset="human_stemcell",
                 test_dataset="pbmc", min_counts=1):
        self.params = ImputationDatasetParams
        self.params.data_dir = data_dir
        self.params.random_seed = random_seed
        self.params.min_counts = min_counts
        self.params.train_dataset = train_dataset
        self.params.test_dataset = test_dataset
        self.params.gpu = gpu
        self.params.filetype = filetype

    def download_all_data(self):

        gene_class = ["pbmc_data", "mouse_brain_data", "mouse_embryo_data", "human_stemcell_data"]

        url = {
            "pbmc_data": "https://www.dropbox.com/s/brj3orsjbhnhawa/5k.zip?dl=0",
            "mouse_embryo_data": "https://www.dropbox.com/s/8ftx1bydoy7kn6p/GSE65525.zip?dl=0",
            "mouse_brain_data": "https://www.dropbox.com/s/zzpotaayy2i29hk/neuron_10k.zip?dl=0",
            "human_stemcell_data": "https://www.dropbox.com/s/g2qua2j3rqcngn6/GSE75748.zip?dl=0"
        }

        file_name = {
            "pbmc_data": "5k.zip?dl=0",
            "mouse_embryo_data": "GSE65525.zip?dl=0",
            "mouse_brain_data": "neuron_10k.zip?dl=0",
            "human_stemcell_data": "GSE75748.zip?dl=0"
        }

        dl_files = {
            "pbmc_data": "5k_*",
            "mouse_embryo_data": "GSE65525",
            "mouse_brain_data": "neuron*",
            "human_stemcell_data": "GSE75748"
        }

        dataset_to_file = {
            "pbmc_data":
            "5k_pbmc_protein_v3_filtered_feature_bc_matrix.h5",
            "mouse_embryo_data":
            list(
                map(lambda x: "GSE65525/" + x, [
                    "GSM1599494_ES_d0_main.csv", "GSM1599497_ES_d2_LIFminus.csv", "GSM1599498_ES_d4_LIFminus.csv",
                    "GSM1599499_ES_d7_LIFminus.csv"
                ])),
            "mouse_brain_data":
            "neuron_10k_v3_filtered_feature_bc_matrix.h5",
            "human_stemcell_data":
            "GSE75748/GSE75748_sc_time_course_ec.csv.gz"
        }
        self.params.dataset_to_file = dataset_to_file
        if sys.platform != 'win32':
            if not osp.exists(self.params.data_dir):
                os.system("mkdir " + self.params.data_dir)
            if not osp.exists(self.params.data_dir + "/train"):
                os.system("mkdir " + self.params.data_dir + "/train")

            for class_name in gene_class:
                if not any(
                        list(
                            map(osp.exists,
                                glob.glob(self.params.data_dir + "/train/" + class_name + "/" +
                                          dl_files[class_name])))):
                    os.system("mkdir " + self.params.data_dir + "/train/" + class_name)
                    os.system("wget " + url[class_name])  # assumes linux... mac needs to install
                    os.system("unzip " + file_name[class_name])
                    os.system("rm " + file_name[class_name])
                    os.system("mv " + dl_files[class_name] + " " + self.params.data_dir + "/train/" + class_name + "/")
            os.system("cp -r " + self.params.data_dir + "/train/ " + self.params.data_dir + "/test")
        if sys.platform == 'win32':
            if not osp.exists(self.params.data_dir):
                os.system("mkdir " + self.params.data_dir)
            if not osp.exists(self.params.data_dir + "/train"):
                os.mkdir(self.params.data_dir + "/train")
            for class_name in gene_class:
                if not any(
                        list(
                            map(osp.exists,
                                glob.glob(self.params.data_dir + "/train/" + class_name + "/" +
                                          dl_files[class_name])))):
                    os.mkdir(self.params.data_dir + "/train/" + class_name)
                    os.system("curl " + url[class_name])
                    os.system("tar -xf " + file_name[class_name])
                    os.system("del -R " + file_name[class_name])
                    os.system("move " + dl_files[class_name] + " " + self.params.data_dir + "/train/" + class_name +
                              "/")
            os.system("copy /r " + self.params.data_dir + "/train/ " + self.params.data_dir + "/test")

    def is_complete(self):
        # check whether data is complete or not
        check = [
            self.params.data_dir + "/train",
            self.params.data_dir + "/test",
        ]

        for i in check:
            if not osp.exists(i):
                print("file {} doesn't exist".format(i))
                return False
        return True

    def load_data(self, model_params, model='GraphSCI'):
        # Load data from existing h5ad files, or download files and load data.
        if self.is_complete():
            pass
        else:
            self.download_all_data()
            assert self.is_complete()

        data_dict = load_imputation_data_internal(self.params, model_params, model=model)
        self.params.num_cells = data_dict['num_cells']
        self.params.num_genes = data_dict['num_genes']
        self.params.train_data = data_dict['train_data']
        self.params.test_data = data_dict['test_data']
        self.params.adata = data_dict['adata']

        if model == 'GraphSCI':
            self.params.train_data_raw = data_dict['train_data_raw']
            self.params.test_data_raw = data_dict['test_data_raw']
            self.params.adj_train = data_dict['adj_train']
            self.params.adj_test = data_dict['adj_test']
            self.params.adj_train_false = data_dict['adj_train_false']
            self.params.adj_norm_train = data_dict['adj_norm_train']
            self.params.adj_norm_test = data_dict['adj_norm_test']
            self.params.size_factors = data_dict['size_factors']
            self.params.train_size_factors = data_dict['train_size_factors']
            self.params.test_size_factors = data_dict['test_size_factors']
            self.params.train_size_factors = data_dict['train_size_factors']
            self.params.test_size_factors = data_dict['test_size_factors']
            self.params.test_idx = data_dict['test_idx']
        if model == 'DeepImpute':
            self.params.X_train = self.params.train_data[0]
            self.params.Y_train = self.params.train_data[1]
            self.params.X_test = self.params.test_data[0]
            self.params.Y_test = self.params.test_data[1]
            self.params.inputgenes = data_dict['predictors']
            self.params.targetgenes = data_dict['targets']
            self.params.total_counts = data_dict['total_counts']
            self.params.true_counts = data_dict['true_counts']
            self.params.genes_to_impute = data_dict['genes_to_impute']
        if model == 'scGNN':
            self.params.genelist = data_dict['genelist']
            self.params.celllist = data_dict['celllist']
            self.params.test_idx = data_dict['test_idx']

        return self
