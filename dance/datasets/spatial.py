import os
import os.path as osp
import warnings
from pprint import pformat

import cv2
import pandas as pd
import scanpy as sc

from dance import logger
from dance.data import Data
from dance.datasets.base import BaseDataset
from dance.registers import register_dataset
from dance.utils.download import download_file, download_unzip, unzip_file

IGNORED_FILES = ["readme.txt"]

cellDeconvo_dataset = {
    "CARD_synthetic": "https://www.dropbox.com/sh/v0vpv0jsnfexj7f/AADpizLGOrF7M8EesDihgbBla?dl=1",
    "GSE174746": "https://www.dropbox.com/sh/spfv06yfttetrab/AAAgORS6ocyoZEyxiRYKTymCa?dl=1",
    "SPOTLight_synthetic": "https://www.dropbox.com/sh/p1tfb0xe1yl2zpe/AAB6cF-BsdJcHToet_C-AlXAa?dl=1",
    "human PDAC": "https://www.dropbox.com/sh/9py6hk9j1ygyprh/AAAOKTo-TE_eX4JJg0HIFfZ7a?dl=1",
    "mouse brain 1": "https://www.dropbox.com/sh/e2nl247v1jrd7h8/AAC1IUlk_3vXUvfk2fv9L2D3a?dl=1",
    "toy1": "https://www.dropbox.com/sh/quvjz6pzltio43u/AAC8vd8-H-4S58-b1pGz3DLRa?dl=1",
    "toy2": "https://www.dropbox.com/sh/eqkcm344p5d1akr/AAAPs0Z0S7yFC5ML8Kcd5eU9a?dl=1",
}


@register_dataset("spatiallibd")
class SpatialLIBDDataset(BaseDataset):

    _DISPLAY_ATTRS = ("data_id", )
    url_dict = {
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

    def __init__(self, root=".", full_download=False, data_id="151673", data_dir="data/spot"):
        super().__init__(root, full_download)

        self.data_id = data_id
        self.data_dir = data_dir + "/{}".format(data_id)

    def download_all(self):
        logger.info(f"All data includes {len(self.url_dict)} datasets: {list(self.url_dict)}")
        _data_id = self.data_id
        for data_id in self.url_dict:
            self.data_id = data_id
            self.download()
        self.data_id = _data_id

    def is_complete_all(self):
        _data_id = self.data_id
        for data_id in self.url_dict:
            self.data_id = data_id
            if not self.is_complete():
                self.data_id = _data_id
                return False
        self.data_id = _data_id
        return True

    def download(self):
        out_path = osp.join(self.data_dir, f"{self.data_id}.zip")
        if download_file(self.url_dict[self.data_id], out_path):
            unzip_file(out_path, self.data_dir)

    def is_complete(self):
        check = [
            osp.join(self.data_dir, f"{self.data_id}_raw_feature_bc_matrix.h5"),  # expression
            osp.join(self.data_dir, f"{self.data_id}_full_image.tif"),  # histology
            osp.join(self.data_dir, "tissue_positions_list.txt"),  # positions
        ]

        for i in check:
            if not os.path.exists(i):
                logger.info(f"lack {i}")
                return False

        return True

    def _load_raw_data(self):
        image_path = osp.join(self.data_dir, f"{self.data_id}_full_image.tif")
        data_path = osp.join(self.data_dir, f"{self.data_id}_raw_feature_bc_matrix.h5")
        spatial_path = osp.join(self.data_dir, "tissue_positions_list.txt")
        meta_path = osp.join(self.data_dir, "cluster_labels.csv")

        logger.info(f"Loading image data from {image_path}")
        img = cv2.imread(image_path)

        logger.info(f"Loading expression data from {data_path}")
        adata = sc.read_10x_h5(data_path)

        logger.info(f"Loading spatial info from {spatial_path}")
        spatial = pd.read_csv(spatial_path, header=None, index_col=0).loc[adata.obs_names]

        logger.info(f"Loading label info from {meta_path}")
        meta_df = pd.read_csv(meta_path)

        # Restrict to captured spots
        indicator = spatial[1].values == 1
        adata = adata[indicator]
        spatial = spatial.iloc[indicator]

        # Prepare spatial info tables
        xy = spatial[[2, 3]].rename(columns={2: "x", 3: "y"})
        xy_pixel = spatial[[4, 5]].rename(columns={4: "x_pixel", 5: "y_pixel"})

        # Prepare meta data and create a column with indexed label info
        label_classes = {j: i for i, j in enumerate(meta_df["ground_truth"].unique())}
        meta_df["label"] = list(map(label_classes.get, meta_df["ground_truth"]))

        return img, adata, xy, xy_pixel, meta_df

    def _raw_to_dance(self, raw_data):
        img, adata, xy, xy_pixel, meta_df = raw_data
        adata.var_names_make_unique()

        adata.obs = meta_df.set_index(adata.obs_names)
        adata.obsm["spatial"] = xy.set_index(adata.obs_names)
        adata.obsm["spatial_pixel"] = xy_pixel.set_index(adata.obs_names)
        adata.uns["image"] = img

        data = Data(adata, train_size="all")
        return data


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
