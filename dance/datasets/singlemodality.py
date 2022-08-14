import argparse
import glob
import os
import os.path as osp
import sys
from dataclasses import dataclass

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset

from dance.data import download_file, download_unzip
from dance.transforms.preprocess import (get_map_dict, load_actinn_data, load_annotation_data_internal,
                                         load_annotation_test_data, load_imputation_data_internal, load_svm_data,
                                         prepare_data_celltypist, splitCommonAnnData)


@dataclass
class CellTypeDatasetParams:
    random_seed = None
    dense_dim = None
    proj_path = None
    train_dataset = None
    test_dataset = None
    species = None
    tissue = None
    score = None
    gpu = None
    evaluate = None
    train_dir = None
    test_dir = None
    statistics_path = None
    map_path = None
    filetype = None
    threshold = None
    exclude_rate = None
    test_rate = None


class CellTypeDataset():

    def __init__(self, data_type="scdeepsort", random_seed=10, proj_path="./", dense_dim=None, train_dataset=None,
                 test_dataset=None, species=None, tissue=None, score=True, gpu=None, evaluate=None, train_dir=None,
                 test_dir=None, statistics_path=None, map_path=None, filetype=None, threshold=None, exclude_rate=None,
                 test_rate=None, data_dir="./", train_set=None, train_label=None, test_set=None, test_label=None,
                 X_celltypist=None, labels_celltypist=None, genes_celltypist=None, transpose_input_celltypist=False,
                 singlecellnet_type="Lung", svm_settings=None):
        self.data_dir = data_dir
        self.data_type = data_type
        if data_type == "scdeepsort":
            self.download_pretrained_data()
        self.params = CellTypeDatasetParams()
        self.params.random_seed = random_seed
        self.params.dense_dim = dense_dim
        self.params.proj_path = proj_path
        self.params.train_dataset = train_dataset
        self.params.test_dataset = test_dataset
        self.params.species = species
        self.params.tissue = tissue
        self.params.score = score
        self.params.gpu = gpu
        self.params.evaluate = evaluate
        self.params.train_dir = train_dir
        self.params.test_dir = test_dir
        self.params.statistics_path = statistics_path
        self.params.map_path = map_path
        self.params.filetype = filetype
        self.params.threshold = threshold
        self.params.exclude_rate = exclude_rate
        self.params.test_rate = test_rate
        self.train_set = train_set
        self.train_label = train_label
        self.test_set = test_set
        self.test_label = test_label
        self.X_celltypist = X_celltypist
        self.labels_celltypist = labels_celltypist
        self.genes_celltypist = genes_celltypist
        self.transpose_input_celltypist = transpose_input_celltypist
        self.singlecellnet_type = singlecellnet_type

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
            "train_mouse_Brain753_celltype":
            "https://www.dropbox.com/s/x2katwk93z06sgw/mouse_Brain753_celltype.csv?dl=1",
            "train_mouse_Brain753_data": "https://www.dropbox.com/s/3f3wbplgo3xa4ww/mouse_Brain753_data.csv?dl=1",
            "train_mouse_Brain3285_celltype":
            "https://www.dropbox.com/s/ozsobozk3ihkrqg/mouse_Brain3285_celltype.csv?dl=1",
            "train_mouse_Brain3285_data": "https://www.dropbox.com/s/zjrloejx8iqdqsa/mouse_Brain3285_data.csv?dl=1",
            "train_mouse_Kidney4682_celltype":
            "https://www.dropbox.com/s/3plrve7g9v428ec/mouse_Kidney4682_celltype.csv?dl=1",
            "train_mouse_Kidney4682_data": "https://www.dropbox.com/s/olf5nirtieu1ikq/mouse_Kidney4682_data.csv?dl=1",
            "train_mouse_Spleen1970_celltype":
            "https://www.dropbox.com/s/3ea64vk546fjxvr/mouse_Spleen1970_celltype.csv?dl=1",
            "train_mouse_Spleen1970_data": "https://www.dropbox.com/s/c4te0fr1qicqki8/mouse_Spleen1970_data.csv?dl=1",
            "test_mouse_Brain2695_celltype":
            "https://www.dropbox.com/s/gh72dk7i0p7fggu/mouse_Brain2695_celltype.csv?dl=1",
            "test_mouse_Brain2695_data": "https://www.dropbox.com/s/ufianih66xjqxdu/mouse_Brain2695_data.csv?dl=1",
            "test_mouse_Kidney203_celltype":
            "https://www.dropbox.com/s/t4eyaig889qdiz2/mouse_Kidney203_celltype.csv?dl=1",
            "test_mouse_Kidney203_data": "https://www.dropbox.com/s/kmos1ceubumgmpj/mouse_Kidney203_data.csv?dl=1",
            "test_mouse_Spleen1759_celltype":
            "https://www.dropbox.com/s/gczehvgai873mhb/mouse_Spleen1759_celltype.csv?dl=1",
            "test_mouse_Spleen1759_data": "https://www.dropbox.com/s/fl8t7rbo5dmznvq/mouse_Spleen1759_data.csv?dl=1",
        }

        # Download training and testing data
        for name, url in urls.items():
            filename = url.split("/")[-1].split("?")[0]
            filepath = osp.join(self.data_dir, *name.split("_")[:2], filename)
            download_file(url, filepath)

        if download_map:
            # Download mapping data
            download_unzip("https://www.dropbox.com/sh/hw1189sgm0kfrts/AAAapYOblLApqygZ-lGo_70-a?dl=1",
                           osp.join(self.data_dir, "map"))

        if download_pretrained:
            # Download pretrained stats data
            download_unzip("https://www.dropbox.com/sh/s2cxcrzl2ama9zp/AACKwiYtS8hbOOudQLIMDvXUa?dl=1",
                           osp.join(self.data_dir, "pretrained"))

    def download_example_data(self):
        # download example data
        os.system("wget https://www.dropbox.com/s/5lbdytwpip0kl58/example.zip?dl=0")
        os.system(f"mv example.zip?dl=0 {self.data_dir}")
        os.system(f"unzip {self.data_dir} /example.zip?dl=0")
        os.system(f"mv {self.data_dir}/example/* {self.data_dir}")
        os.system(f"rm {self.data_dir}/example.zip?dl=0")
        os.system(f"rm -r {self.data_dir}/example")

    def download_pretrained_data(self):
        """Download pretrained model and label maps."""
        os.system("wget https://www.dropbox.com/s/i5mufqwc1hy97m4/pretrained.zip?dl=0")
        os.system(f"mv pretrained.zip?dl=0 {self.data_dir}")
        os.system(f"unzip {self.data_dir}/pretrained.zip?dl=0")
        os.system(f"rm {self.data_dir}/pretrained.zip?dl=0")

    def download_actinn_data(self):
        """Download pretrained model and label maps."""
        os.system("wget https://www.dropbox.com/s/me2zu2rok9twcjg/actinn_data.zip?dl=0")
        os.system("unzip actinn_data.zip?dl=0")
        os.system("rm actinn_data.zip?dl=0")

    def download_singlecellnet_data(self):
        """Download pretrained model and label maps."""
        print("Downloading Training Data from Bladder...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adBladder_TabSen_101320.h5ad")
        print("Downloading Training Data from Fat...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adFat_TabSen_101320.h5ad")
        print("Downloading Training Data from Heart...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adHeart_TabSen_101320.h5ad")
        print("Downloading Training Data from Kidney...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adKidney_TabSen_101320.h5ad")
        print("Downloading Training Data from Large Intestine...")
        os.system(
            "wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adL_Intestine_TabSen_101320.h5ad")
        print("Downloading Training Data from Lung...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adLung_TabSen_100920.h5ad")
        print("Downloading Training Data from Mammary Gland...")
        os.system(
            "wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adMammary_Gland_TabSen_101320.h5ad")
        print("Downloading Training Data from Marrow...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adMarrow_TabSen_101320.h5ad")
        print("Downloading Training Data from Pancreas...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adPancreas_TabSen_101320.h5ad")
        print("Downloading Training Data from Skeletal Muscle...")
        os.system(
            "wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adSkel_Muscle_TabSen_101320.h5ad")
        print("Downloading Training Data from Skin...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adSkin_TabSen_101320.h5ad")
        print("Downloading Training Data from Trachea...")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/training/adTrachea_TabSen_101320.h5ad")

    def download_singlecellnet_lung_example_data(self):
        print("Downloading lung example data...")
        os.system(
            "wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/query/GSE124872_raw_counts_single_cell.mtx.gz")
        os.system("gzip -d  GSE124872_raw_counts_single_cell.mtx.gz")
        os.system(
            "wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/query/GSE124872_Angelidis_2018_metadata.csv")
        os.system("wget https://cnobjects.s3.amazonaws.com/singleCellNet/pySCN/query/genes.csv")

    def is_complete(self):
        """Check if data is complete."""
        check = [
            osp.join(self.data_dir, "train"),
            osp.join(self.data_dir, "test"),
            osp.join(self.data_dir, "pretrained")
        ]

        for i in check:
            if not os.path.exists(i):
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
            if not os.path.exists(file_i):
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
            if not os.path.exists(file):
                print(f"file {name} doesn't exist")
                return False
        # TODO: check pretrained data
        return True

    def is_actinn_complete(self):
        # check whether data is complete or not
        check = ["test_data"]
        for i in check:
            if not os.path.exists(i):
                print(f"file {i} doesn't exist")
                return False
        return True

    def is_singlecellnet_complete(self):
        # check whether data is complete or not
        check = [
            "Bladder",
            "Fat",
            "Heart",
            "Kidney",
            "L_Intestine",
            "Lung",
            "Mammary_Gland",
            "Marrow",
            "Pancreas",
            "Skel_Muscle",
            "Skin",
            "Trachea",
        ]
        for i in check:
            if i == "Lung":
                file_i = "adLung_TabSen_100920.h5ad"
            else:
                file_i = "ad" + i + "_TabSen_101320.h5ad"
            if not os.path.exists(file_i):
                print(f"file {i} doesn't exist")
                return False
        return True

    def load_data(self):
        # Load data from existing h5ad files, or download files and load data.
        if self.data_type == "scdeepsort" or self.data_type == "scdeepsort_exp":
            if self.is_complete():
                pass
            else:
                if self.data_type == "scdeepsort":
                    self.download_all_data()
                if self.data_type == "scdeepsort_exp":
                    self.download_benchmark_data()
                assert self.is_complete()

            (
                self.num_cells,
                self.num_genes,
                self.num_labels,
                self.graph,
                self.train_ids,
                self.test_ids,
                self.labels,
            ) = load_annotation_data_internal(self.params)

            if self.params.score:
                (
                    self.total_cell_test,
                    self.num_genes_test,
                    self.num_labels_test,
                    self.id2label_test,
                    self.test_dict,
                    self.test_label_dict,
                    self.time_used_test,
                ) = load_annotation_test_data(self.params)
            else:
                (
                    self.total_cell_test,
                    self.num_genes_test,
                    self.num_labels_test,
                    self.id2label_test,
                    self.test_dict,
                    self.time_used_test,
                ) = load_annotation_test_data(self.params)

        if self.data_type == "svm":
            if self.is_complete():
                pass
            else:
                if self.data_type == "svm":
                    self.download_benchmark_data()
                if self.data_type == "svm_exp":
                    self.download_example_data()
                assert self.is_complete()

            (
                self.svm_num_labels,
                self.svm_train_labels,
                self.svm_train_cell_feat,
                self.svm_map_dict,
                self.svm_id2label,
                self.svm_test_label_dict,
                self.svm_test_feat_dict,
                self.svm_test_cell_id_dict,
            ) = load_svm_data(self.params)

        if self.data_type == "actinn":
            self.download_benchmark_data()
            (
                self.train_set,
                self.train_label,
                self.test_set,
                self.test_label,
                self.barcode,
                self.label_to_type_dict,
            ) = load_actinn_data(self.train_set, self.train_label, self.test_set, self.test_label)

        if self.data_type == "celltypist":
            self.download_benchmark_data(download_pretrained=False)
            self.map_dict = get_map_dict(self.params.map_path, self.params.tissue)  # load map
            train_data = pd.read_csv(
                osp.join(self.params.proj_path, self.params.train_dir, self.params.species,
                         self.params.species + "_" + self.params.tissue + str(self.params.train_dataset) + "_data.csv"),
                index_col=0)
            train_celltype = pd.read_csv(
                osp.join(
                    self.params.proj_path, self.params.train_dir, self.params.species,
                    self.params.species + "_" + self.params.tissue + str(self.params.train_dataset) + "_celltype.csv"),
                index_col=1)
            self.train_adata = ad.AnnData(train_data.T, train_celltype)
            test_data = pd.read_csv(
                osp.join(self.params.proj_path, self.params.test_dir, self.params.species,
                         self.params.species + "_" + self.params.tissue + str(self.params.test_dataset) + "_data.csv"),
                index_col=0)
            test_celltype = pd.read_csv(
                osp.join(
                    self.params.proj_path, self.params.test_dir, self.params.species,
                    self.params.species + "_" + self.params.tissue + str(self.params.test_dataset) + "_celltype.csv"),
                index_col=1)
            self.test_adata = ad.AnnData(test_data.T, test_celltype)

        if self.data_type == "singlecellnet_exp":
            if self.is_singlecellnet_complete():
                pass
            else:
                self.download_singlecellnet_data()
                self.download_singlecellnet_lung_example_data()

            if self.singlecellnet_type == "Lung":
                file_i = "adLung_TabSen_100920.h5ad"
            else:
                file_i = "ad" + self.singlecellnet_type + "_TabSen_101320.h5ad"
            adTrain = sc.read(file_i)
            qDatT = sc.read_mtx("GSE124872_raw_counts_single_cell.mtx")
            qDat = qDatT.T
            genes = pd.read_csv("genes.csv")
            qDat.var_names = genes.x
            qMeta = pd.read_csv("GSE124872_Angelidis_2018_metadata.csv")
            qMeta.columns.values[0] = "cellid"
            qMeta.index = qMeta["cellid"]
            qDat.obs = qMeta.copy()
            genesTrain = adTrain.var_names
            genesQuery = qDat.var_names
            cgenes = genesTrain.intersection(genesQuery)
            adTrain1 = adTrain[:, cgenes]
            adQuery = qDat[:, cgenes].copy()
            adQuery = adQuery[adQuery.obs["nGene"] >= 500, :].copy()
            self.expTrain, self.expVal = splitCommonAnnData(adTrain1, ncells=200, dLevel="cell_ontology_class")

        if self.data_type == "singlecellnet":
            self.download_benchmark_data(download_pretrained=False)
            self.map_dict = get_map_dict(self.params.map_path, self.params.tissue)  # load map
            train_data = pd.read_csv(
                osp.join(self.params.proj_path, self.params.train_dir, self.params.species,
                         self.params.species + "_" + self.params.tissue + str(self.params.train_dataset) + "_data.csv"),
                index_col=0)
            train_celltype = pd.read_csv(
                osp.join(
                    self.params.proj_path, self.params.train_dir, self.params.species,
                    self.params.species + "_" + self.params.tissue + str(self.params.train_dataset) + "_celltype.csv"),
                index_col=1)
            self.train_adata = ad.AnnData(train_data.T, train_celltype)
            test_data = pd.read_csv(
                osp.join(self.params.proj_path, self.params.test_dir, self.params.species,
                         self.params.species + "_" + self.params.tissue + str(self.params.test_dataset) + "_data.csv"),
                index_col=0)
            test_celltype = pd.read_csv(
                osp.join(
                    self.params.proj_path, self.params.test_dir, self.params.species,
                    self.params.species + "_" + self.params.tissue + str(self.params.test_dataset) + "_celltype.csv"),
                index_col=1)
            self.test_adata = ad.AnnData(test_data.T, test_celltype)

            if self.is_singlecellnet_complete():
                pass
            else:
                self.download_singlecellnet_data()
                self.download_singlecellnet_lung_example_data()

            if self.singlecellnet_type == "Lung":
                file_i = "adLung_TabSen_100920.h5ad"
            else:
                file_i = "ad" + self.singlecellnet_type + "_TabSen_101320.h5ad"
            adTrain = sc.read(file_i)
            qDatT = sc.read_mtx("GSE124872_raw_counts_single_cell.mtx")
            qDat = qDatT.T
            genes = pd.read_csv("genes.csv")
            qDat.var_names = genes.x
            qMeta = pd.read_csv("GSE124872_Angelidis_2018_metadata.csv")
            qMeta.columns.values[0] = "cellid"
            qMeta.index = qMeta["cellid"]
            qDat.obs = qMeta.copy()
            genesTrain = adTrain.var_names
            genesQuery = qDat.var_names
            cgenes = genesTrain.intersection(genesQuery)
            adTrain1 = adTrain[:, cgenes]
            adQuery = qDat[:, cgenes].copy()
            adQuery = adQuery[adQuery.obs["nGene"] >= 500, :].copy()
            self.expTrain, self.expVal = splitCommonAnnData(adTrain1, ncells=200, dLevel="cell_ontology_class")

        return self


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
        return os.path.exists(os.path.join(self.data_dir, f"{self.dataset}.h5"))

    def load_data(self):
        # Load data from existing h5ad files, or download files and load data.
        if self.is_complete():
            pass
        else:
            self.download_data()
            assert self.is_complete()

        data_mat = h5py.File(f"{self.data_dir}/{self.dataset}.h5", "r")
        self.X = np.array(data_mat["X"])
        self.Y = np.array(data_mat["Y"])
        return self


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

    def __init__(
            self,
            random_seed=10,
            gpu=-1,
            # evaluate = None,
            filetype=None,
            data_dir="data",
            train_dataset="human_stemcell",
            test_dataset="pbmc",
            min_counts=1):
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
            if not os.path.exists(self.params.data_dir):
                os.system("mkdir " + self.params.data_dir)
            if not os.path.exists(self.params.data_dir + "/train"):
                os.system("mkdir " + self.params.data_dir + "/train")

            for class_name in gene_class:
                if not any(
                        list(
                            map(os.path.exists,
                                glob.glob(self.params.data_dir + "/train/" + class_name + "/" +
                                          dl_files[class_name])))):
                    os.system("mkdir " + self.params.data_dir + "/train/" + class_name)
                    os.system("wget " + url[class_name])  # assumes linux... mac needs to install
                    os.system("unzip " + file_name[class_name])
                    os.system("rm " + file_name[class_name])
                    os.system("mv " + dl_files[class_name] + " " + self.params.data_dir + "/train/" + class_name + "/")
            os.system("cp -r " + self.params.data_dir + "/train/ " + self.params.data_dir + "/test")
        if sys.platform == 'win32':
            if not os.path.exists(self.params.data_dir):
                os.system("mkdir " + self.params.data_dir)
            if not os.path.exists(self.params.data_dir + "/train"):
                os.mkdir(self.params.data_dir + "/train")
            for class_name in gene_class:
                if not any(
                        list(
                            map(os.path.exists,
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
            if not os.path.exists(i):
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
