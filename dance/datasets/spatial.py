import os
import os.path as osp
import warnings
from pprint import pformat

import cv2
import pandas as pd
import scanpy as sc

from dance import logger
from dance.data import download_file, download_unzip, unzip_file

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


class SpotDataset:

    def __init__(self, data_id="151673", data_dir="data/spot", build_graph_fn="default"):
        self.data_id = data_id
        self.data_dir = data_dir + "/{}".format(data_id)
        self.data_url = dataset[data_id]
        self._load_data()
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

    def _load_data(self):
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

    def load_data(self):
        adata = self.data
        spatial = adata.obs[["x", "y"]]
        spatial_pixel = adata.obs[["x_pixel", "y_pixel"]]
        image = self.img
        label = adata.obs[["label"]]
        return image, adata, spatial, spatial_pixel, label


class CellTypeDeconvoDatasetLite:

    def __init__(self, data_id="GSE174746", data_dir="data/spatial", build_graph_fn="default"):
        if data_id not in cellDeconvo_dataset:
            raise ValueError(f"Unknown data_id {data_id!r}, available datasets are: {sorted(cellDeconvo_dataset)}")

        self.data_id = data_id
        self.data_dir = osp.join(data_dir, data_id)
        self.data_url = cellDeconvo_dataset[data_id]
        self._load_data()

    def _load_data(self):
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

    def load_data(self, subset_common_celltypes: bool = True):
        """Load raw data.

        Parameters
        ----------
        subset_common_celltypes
            If set to True, then subset both the reference and the real data to contain only cell types that are
            present in both reference and real.

        """
        ref_count = self.data["ref_sc_count"]
        ref_annot = self.data["ref_sc_annot"]
        count_matrix = self.data["mix_count"]
        cell_type_portion = self.data["true_p"]
        if (spatial := self.data.get("spatial_location")) is None:
            spatial = pd.DataFrame(0, index=count_matrix.index, columns=["x", "y"])

        # Obtain cell type info and subset to common cell types between ref and real if needed
        ref_celltypes = set(ref_annot["cellType"].unique().tolist())
        real_celltypes = set(cell_type_portion.columns.tolist())
        logger.info(f"Number of cell types: reference = {len(ref_celltypes)}, real = {len(real_celltypes)}")
        if subset_common_celltypes:
            common_celltypes = sorted(ref_celltypes & real_celltypes)
            logger.info(f"Subsetting to common cell types (n={len(common_celltypes)}):\n{pformat(common_celltypes)}")

            idx = ref_annot[ref_annot["cellType"].isin(common_celltypes)].index
            ref_annot = ref_annot.loc[idx]
            ref_count = ref_count.loc[idx]

            cell_type_portion = cell_type_portion[common_celltypes]

        return ref_count, ref_annot, count_matrix, cell_type_portion, spatial
