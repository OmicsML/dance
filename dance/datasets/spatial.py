import csv
import glob
import os
import os.path as osp
import pickle as pkl
import random
import re
import time as tm
import warnings
from collections import defaultdict
from operator import itemgetter

import anndata
import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rdata
import scanpy as sc
import scipy.sparse
from anndata import AnnData
from scipy.stats import uniform

from dance.data import download_file, download_unzip, unzip_file
from dance.transforms import preprocess

IGNORED_FILES = ["readme.txt"]

dataset = {
    "151510": "https://www.dropbox.com/sh/41h9brsk6my546x/AADa18mkJge-KQRTndRelTpMa?dl=1",
    "151507": "https://www.dropbox.com/sh/m3554vfrdzbwv2c/AACGsFNVKx8rjBgvF7Pcm2L7a?dl=1",
    "151508": "https://www.dropbox.com/sh/tm47u3fre8692zt/AAAJJf8-za_Lpw614ft096qqa?dl=1",
    "151509": "https://www.dropbox.com/sh/hihr7906vyirjet/AACslV5mKIkF2CF5QqE1LE6ya?dl=1",
    "151669": "https://www.dropbox.com/sh/ulw2nnnmgtbswvc/AAC0fT549EwtxKZWWoB89gb4a?dl=1",
    "151670": "https://www.dropbox.com/sh/8fw44zyyjgh0ddc/AAA1asGAmyDiMmvhRmL7pN1Na?dl=1",
    "151671": "https://www.dropbox.com/sh/9g5qzd5ykx2mpk3/AAD3xjx1i2h0RhYBc-Vft6CEa?dl=1",
    "151672": "https://www.dropbox.com/sh/l6519tr280krd4p/AAAWefCSp2iKhVmLgytlyxTta?dl=1",
    "151673": "https://www.dropbox.com/sh/qc64ps6gd64dm0c/AAC_5_mP4AczKj8lORLLKcIba?dl=1",
    "151674": "https://www.dropbox.com/sh/q7io99psd2xuqgw/AABske8dgX_kc1oaDSxuiqjpa?dl=1",
    "151675": "https://www.dropbox.com/sh/uahka2h5klnrzvj/AABe7K0_ewqOcqKUxHebE6qLa?dl=1",
    "151676": "https://www.dropbox.com/sh/jos5jjurezy5zp1/AAB2uaVm3-Us1a4mDkS1Q-iAa?dl=1",
}

cellDeconvo_dataset = {
    "CARD_synthetic": "https://www.dropbox.com/sh/v0vpv0jsnfexj7f/AADpizLGOrF7M8EesDihgbBla?dl=1",
    "GSE174746": "https://www.dropbox.com/sh/spfv06yfttetrab/AAAgORS6ocyoZEyxiRYKTymCa?dl=1",
    "SPOTLight_synthetic": "https://www.dropbox.com/sh/p1tfb0xe1yl2zpe/AAB6cF-BsdJcHToet_C-AlXAa?dl=1",
    "human PDAC": "https://www.dropbox.com/sh/9py6hk9j1ygyprh/AAAOKTo-TE_eX4JJg0HIFfZ7a?dl=1",
    "mouse brain 1": "https://www.dropbox.com/sh/e2nl247v1jrd7h8/AAC1IUlk_3vXUvfk2fv9L2D3a?dl=1",
    "toy1": "https://www.dropbox.com/sh/quvjz6pzltio43u/AAC8vd8-H-4S58-b1pGz3DLRa?dl=1",
    "toy2": "https://www.dropbox.com/sh/eqkcm344p5d1akr/AAAPs0Z0S7yFC5ML8Kcd5eU9a?dl=1",
}

card_simulation_dataset = {
    "sim_noise0_rep1":
    "https://www.dropbox.com/s/aujusznwbq4xa99/sim.pseudo.MOB.n10.cellType6.Mixnoise0.repeat1.RData?dl=1",
    "sim_noise0_rep2":
    "https://www.dropbox.com/s/bjyagojp3opkyxd/sim.pseudo.MOB.n10.cellType6.Mixnoise0.repeat2.RData?dl=1",
    "sim_noise0_rep3":
    "https://www.dropbox.com/s/k9n07ujvqtroj72/sim.pseudo.MOB.n10.cellType6.Mixnoise0.repeat3.RData?dl=1",
    "sim_noise0_rep4":
    "https://www.dropbox.com/s/3m49dq387he776x/sim.pseudo.MOB.n10.cellType6.Mixnoise0.repeat4.RData?dl=1",
    "sim_noise0_rep5":
    "https://www.dropbox.com/s/4wjsl2ids1q16b2/sim.pseudo.MOB.n10.cellType6.Mixnoise0.repeat5.RData?dl=1",
    "sim_noise3_rep1":
    "https://www.dropbox.com/s/z6ehj48q0vcxf13/sim.pseudo.MOB.n10.cellType6.Mixnoise3.repeat1.RData?dl=1",
    "sim_noise3_rep2":
    "https://www.dropbox.com/s/2mzjt4pv1f5ucs2/sim.pseudo.MOB.n10.cellType6.Mixnoise3.repeat2.RData?dl=1",
    "sim_noise3_rep3":
    "https://www.dropbox.com/s/65ixbnu6x65o8ee/sim.pseudo.MOB.n10.cellType6.Mixnoise3.repeat3.RData?dl=1",
    "sim_noise3_rep4":
    "https://www.dropbox.com/s/hrmwoi14wta0ida/sim.pseudo.MOB.n10.cellType6.Mixnoise3.repeat4.RData?dl=1",
    "sim_noise3_rep5":
    "https://www.dropbox.com/s/0txpltfj2p3dz9v/sim.pseudo.MOB.n10.cellType6.Mixnoise3.repeat5.RData?dl=1",
}


class SpotDataset:

    def __init__(self, data_id="151673", data_dir="data/spot", build_graph_fn="default"):
        self.data_id = data_id
        self.data_dir = data_dir + "/{}".format(data_id)
        self.data_url = dataset[data_id]
        self.load_data()
        self.adj = None

    def get_all_data(self):
        # provide an interface to get all data at one time
        print("All data includes {} datasets: {}".format(len(dataset), ",".join(dataset.keys())))
        res = {}
        for each_dataset in dataset.keys():
            res[each_dataset] = SpotDataset(each_dataset)
        return res

    def download_data(self):
        # judge whether a file exists or not
        isdownload = download_file(self.data_url, self.data_dir + "/{}.zip".format(self.data_id))
        if isdownload:
            unzip_file(self.data_dir + "/{}.zip".format(self.data_id), self.data_dir + "/")
        return self

    def is_complete(self):
        # data.h5ad
        # histology.tif
        # positions.txt
        # judge whether data is complete or not
        check = [
            self.data_dir + "/{}_raw_feature_bc_matrix.h5".format(self.data_id),
            self.data_dir + "/{}_full_image.tif".format(self.data_id), self.data_dir + "/tissue_positions_list.txt"
        ]

        for i in check:
            if not os.path.exists(i):
                print("lack {}".format(i))
                return False
        return True

    def load_data(self):
        if self.is_complete():
            pass
        else:
            self.download_data()
        self.data = sc.read_10x_h5(self.data_dir + "/{}_raw_feature_bc_matrix.h5".format(self.data_id))
        self.img = cv2.imread(self.data_dir + "/{}_full_image.tif".format(self.data_id))
        label = pd.read_csv(self.data_dir + "/cluster_labels.csv")
        classes = {layer_class: idx for idx, layer_class in enumerate(set(label["ground_truth"].tolist()))}
        self.spatial = pd.read_csv(self.data_dir + "/tissue_positions_list.txt", sep=",", header=None, na_filter=False,
                                   index_col=0)
        self.data.obs["x1"] = self.spatial[1]
        self.data.obs["x2"] = self.spatial[2]
        self.data.obs["x3"] = self.spatial[3]
        self.data.obs["x4"] = self.spatial[4]
        self.data.obs["x5"] = self.spatial[5]
        self.data.obs["x"] = self.data.obs["x2"]
        self.data.obs["y"] = self.data.obs["x3"]
        self.data.obs["x_pixel"] = self.data.obs["x4"]
        self.data.obs["y_pixel"] = self.data.obs["x5"]

        self.data = self.data[self.data.obs["x1"] == 1]
        self.data.var_names = [i.upper() for i in list(self.data.var_names)]
        self.data.var["genename"] = self.data.var.index.astype("str")
        self.data.obs["label"] = list(map(lambda x: classes[x], label["ground_truth"].tolist()))
        self.data.obs["ground_truth"] = label["ground_truth"].tolist()
        return self


class CellTypeDeconvoDataset:

    def __init__(self, data_id="toy1", data_dir="data/spatial", build_graph_fn="default"):
        self.data_id = data_id
        self.data_dir = data_dir + "/{}".format(data_id)
        self.data_url = cellDeconvo_dataset[data_id]
        self.load_data()
        self.adj = None

    def get_all_data(self):
        # provide an interface to get all data at one time
        print("All data includes {} cellDeconvo datasets: {}".format(len(cellDeconvo_dataset),
                                                                     ",".join(cellDeconvo_dataset.keys())))
        res = {}
        for each_dataset in cellDeconvo_dataset.keys():
            res[each_dataset] = cellDeconvo_dataset(each_dataset)
        return res

    def download_data(self):
        # judge whether a file exists or not
        isdownload = download_file(self.data_url, self.data_dir + "/{}.zip".format(self.data_id))
        if isdownload:
            unzip_file(self.data_dir + "/{}.zip".format(self.data_id), self.data_dir)
        return self

    def is_complete(self):
        check = [self.data_dir + "/mix_count.*", self.data_dir + "/ref_sc_count.*"]

        for i in check:
            #if not os.path.exists(i):
            if not glob.glob(i):
                print("lack {}".format(i))
                return False
        return True

    def load_data(self):
        if self.is_complete():
            pass
        else:
            self.download_data()

        self.data = {}
        files = os.listdir(self.data_dir + "/")
        filenames = [f.split(".")[0] for f in files]
        extensions = [f.split(".")[1] for f in files]
        for f in files:
            DataPath = self.data_dir + "/" + f
            filename = f.split(".")[0]
            ext = f.split(".")[1]
            if ext == "csv":
                data = pd.read_csv(DataPath, header=0, index_col=0)
                self.data[filename] = data
            elif ext == "h5ad":
                data = sc.read_h5ad(DataPath)
                self.data[filename] = data
                self.data[filename + "_annot"] = data.obs
            else:
                print("unsupported file type. Please use csv or h5ad file types.")

        print("load data succesfully....")

        return self


class CellTypeDeconvoDatasetLite:

    def __init__(self, data_id="GSE174746", data_dir="data/spatial", build_graph_fn="default"):
        if data_id not in cellDeconvo_dataset:
            raise ValueError(f"Unknown data_id {data_id!r}, available datasets are: {sorted(cellDeconvo_dataset)}")

        self.data_id = data_id
        self.data_dir = osp.join(data_dir, data_id)
        self.data_url = cellDeconvo_dataset[data_id]
        self.load_data()

    def load_data(self):
        if not osp.exists(self.data_dir):
            download_unzip(self.data_url, self.data_dir)

        self.data = {}
        for f in os.listdir(self.data_dir):
            filepath = osp.join(self.data_dir, f)
            filename, ext = osp.splitext(f)
            if f in IGNORED_FILES:
                continue
            elif ext == ".csv":
                self.data[filename] = pd.read_csv(filepath, header=0, index_col=0)
            elif ext == ".h5ad":
                self.data[filename] = sc.read_h5ad(filepath).to_df()
            else:
                warnings.warn(f"Unsupported file type {ext!r}. Use csv or h5ad file types.")


class CARDSimulationRDataset:
    ref_sc_count_url: str = "https://www.dropbox.com/s/wchoppxcsulk8ev/split2_ref_sc_count.h5ad?dl=1"
    ref_sc_annot_url: str = "https://www.dropbox.com/s/irpvco2ffisvxvk/split2_ref_sc_annot.csv?dl=1"

    def __init__(self, data_id="sim_noise0_rep1", data_dir="data/spatial/card_simulation", build_graph_fn="default"):
        self.data_id = data_id
        self.data_dir = osp.join(data_dir, data_id)
        self.data_path = osp.join(data_dir, f"{data_id}.RData")
        self.ref_sc_count_path = osp.join(data_dir, "ref_sc_count.h5ad")
        self.ref_sc_annot_path = osp.join(data_dir, "ref_sc_annot.csv")
        self.data_url = card_simulation_dataset[data_id]
        self.load_data()
        self.adj = None

    def get_all_data(self):
        # TODO: make classmethod, make data url dict as class attrs
        dataset_info = "\n\t".join(list(card_simulation_dataset))
        print(f"Total of {(len(card_simulation_dataset))} datasets:\n{dataset_info}")
        return {i: CARDSimulationRDataset(i) for i in card_simulation_dataset}

    def download_data(self):
        download_file(self.ref_sc_count_url, self.ref_sc_count_path)
        download_file(self.ref_sc_annot_url, self.ref_sc_annot_path)
        download_file(self.data_url, self.data_path)

    def is_complete(self):
        check = [self.data_path, self.ref_sc_count_path, self.ref_sc_annot_path]
        return all(map(osp.exists, check))

    def load_data(self):
        if not self.is_complete():
            self.download_data()

        raw = rdata.conversion.convert(rdata.parser.parse_file(self.data_path))["spatial.pseudo"]
        ref_sc_count = anndata.read_h5ad(self.ref_sc_count_path).to_df().T
        ref_sc_annot = pd.read_csv(self.ref_sc_annot_path, index_col=0)
        spatial_count = raw["pseudo.data"].to_pandas().T
        spatial_location = (spatial_count.reset_index()["dim_1"].str.split("x", expand=True).set_index(
            spatial_count.index).rename({
                0: "x",
                1: "y"
            }, axis=1).astype(float))
        true_p = raw["true.p"].to_pandas()

        # TODO: directly save as attrs instead of a dict?
        self.data = {
            "ref_sc_count": ref_sc_count,
            "ref_sc_annot": ref_sc_annot,
            "spatial_count": spatial_count,
            "spatial_location": spatial_location,
            "true_p": true_p,
        }

        return self
