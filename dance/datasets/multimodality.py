import os
import os.path as osp
import pickle
from abc import ABC

import anndata as ad
import mudata as md
import numpy as np
import scanpy as sc

from dance import logger
from dance.data import Data
from dance.datasets.base import BaseDataset
from dance.transforms.preprocess import lsiTransformer
from dance.typing import List
from dance.utils.download import download_file, unzip_file


class MultiModalityDataset(BaseDataset, ABC):

    TASK = "N/A"
    URL_DICT = {}
    SUBTASK_NAME_MAP = {}
    AVAILABLE_DATA = []

    def __init__(self, subtask, root="./data"):
        assert subtask in self.AVAILABLE_DATA, f"Undefined subtask {subtask!r}."
        assert self.TASK in ["predict_modality", "match_modality", "joint_embedding"]

        self.subtask = self.SUBTASK_NAME_MAP.get(subtask, subtask)
        self.data_url = self.URL_DICT[self.subtask]

        super().__init__(root=root, full_download=False)

    def download(self):
        self.download_data()

    def download_data(self):
        download_file(self.data_url, osp.join(self.root, f"{self.subtask}.zip"))
        unzip_file(osp.join(self.root, f"{self.subtask}.zip"), self.root)

    def download_pathway(self):
        download_file("https://www.dropbox.com/s/uqoakpalr3albiq/h.all.v7.4.entrez.gmt?dl=1",
                      osp.join(self.root, "h.all.v7.4.entrez.gmt"))
        download_file("https://www.dropbox.com/s/yjrcsd2rpmahmfo/h.all.v7.4.symbols.gmt?dl=1",
                      osp.join(self.root, "h.all.v7.4.symbols.gmt"))

    @property
    def data_paths(self) -> List[str]:
        if self.TASK == "joint_embedding":
            mod = "adt" if "cite" in self.subtask else "atac"
            meta = "cite" if "cite" in self.subtask else "multiome"
            paths = [
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_mod2.h5ad"),
                osp.join(self.root, self.subtask, f"{meta}_gex_processed_training.h5ad"),
                osp.join(self.root, self.subtask, f"{meta}_{mod}_processed_training.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_solution.h5ad"),
            ]
        elif self.TASK == "predict_modality":
            paths = [
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod2.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod2.h5ad")
            ]
            if self.subtask == "10k_pbmc":
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_test_mod2.h5ad")
                ]
            if self.subtask == "pbmc_cite":
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_test_mod2.h5ad")
                ]
            if self.subtask.startswith("5k_pbmc"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_test_mod2.h5ad")
                ]
            if self.subtask.startswith("openproblems_2022"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_test_mod2.h5ad")
                ]
            if self.subtask.startswith("GSE127064"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_test_mod2.h5ad")
                ]
            if self.subtask.startswith("GSE117089"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_test_mod2.h5ad")
                ]
            if self.subtask.startswith("GSE140203"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_test_mod2.h5ad")
                ]
        elif self.TASK == "match_modality":
            paths = [
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod2.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_sol.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod2.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_sol.h5ad"),
            ]
        return paths

    def is_complete(self) -> bool:
        return all(map(osp.exists, self.data_paths))

    def _load_raw_data(self) -> List[ad.AnnData]:
        modalities = []
        for mod_path in self.data_paths:
            logger.info(f"Loading {mod_path}")
            modalities.append(ad.read_h5ad(mod_path))
        return modalities


class ModalityPredictionDataset(MultiModalityDataset):

    TASK = "predict_modality"
    URL_DICT = {
        "openproblems_bmmc_cite_phase2_mod2":
        "https://www.dropbox.com/s/snh8knscnlcq4um/openproblems_bmmc_cite_phase2_mod2.zip?dl=1",
        "openproblems_bmmc_cite_phase2_rna":
        "https://www.dropbox.com/s/xbfyhv830u9pupv/openproblems_bmmc_cite_phase2_rna.zip?dl=1",
        "openproblems_bmmc_multiome_phase2_mod2":
        "https://www.dropbox.com/s/p9ve2ljyy4yqna4/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1",
        "openproblems_bmmc_multiome_phase2_rna":
        "https://www.dropbox.com/s/cz60vp7bwapz0kw/openproblems_bmmc_multiome_phase2_rna.zip?dl=1",
        "openproblems_bmmc_cite_phase2_rna_subset":
        "https://www.dropbox.com/s/veytldxkgzyoa8j/openproblems_bmmc_cite_phase2_rna_subset.zip?dl=1",
        "5k_pbmc":
        "https://www.dropbox.com/scl/fi/uoyis946glh0oo7g833qj/5k_pbmc.zip?rlkey=mw9cvqq7e12iowfbr9rp7av5u&dl=1",
        "5k_pbmc_subset":
        "https://www.dropbox.com/scl/fi/pykqc9zyt1fjypnjf4m1l/5k_pbmc_subset.zip?rlkey=brkmnqhfz5yl9axiuu0f8gmxy&dl=1",
        "10k_pbmc":
        "https://www.dropbox.com/scl/fi/npz3n36d3w089creppph2/10k_pbmc.zip?rlkey=6yyv61omv2rw7sqqmfp6u7m1s&dl=1",
        "pbmc_cite":
        "https://www.dropbox.com/scl/fi/8yvel9lu2f4pbemjeihzq/pbmc_cite.zip?rlkey=5f5jpjy1fcg14hwzot0hot7xd&dl=1",
        "openproblems_2022_multi_atac2gex":
        "https://www.dropbox.com/scl/fi/4ynxepu306g3k6vqpi3aw/openproblems_2022_multi_atac2gex.zip?rlkey=2mq5vjnsh26gg5zgq9d85ikcp&dl=1",
        "openproblems_2022_cite_gex2adt":
        "https://www.dropbox.com/scl/fi/dalt3qxwe440107ihjbpy/openproblems_2022_cite_gex2adt.zip?rlkey=ps1fvcr622vhibc1wc1umfdaw&dl=1",
        "GSE127064_AdBrain_gex2atac":
        "https://www.dropbox.com/scl/fi/4ybsx6pgiuy6j9m0y92ly/GSE127064_AdBrain_gex2atac.zip?rlkey=6a5u7p7xr2dqsoduflzxjluja&dl=1",
        "GSE127064_p0Brain_gex2atac":
        "https://www.dropbox.com/scl/fi/k4p3nkkqq56ev6ljyo5se/GSE127064_p0Brain_gex2atac.zip?rlkey=y7kayqmk2l72jjogzlvfxtl74&dl=1",
        "GSE117089_mouse_gex2atac":
        "https://www.dropbox.com/scl/fi/egktuwiognr06xebeuouk/GSE117089_mouse_gex2atac.zip?rlkey=jadp3hlopc3112lmxe6nz5cd1&dl=1",
        "GSE117089_A549_gex2atac":
        "https://www.dropbox.com/scl/fi/b7evc2n5ih5o3xxwcd7uq/GSE117089_A549_gex2atac.zip?rlkey=b5o0ykptfodim59qwnu2m89fh&dl=1",
        "GSE117089_sciCAR_gex2atac":
        "https://www.dropbox.com/scl/fi/juibpvmtv2otvfsq1xyr7/GSE117089_sciCAR_gex2atac.zip?rlkey=qcdbfqsuhab56bc553cwm78gc&dl=1",
        "GSE140203_3T3_HG19_atac2gex":
        "https://www.dropbox.com/scl/fi/v1vbypz87t1rz012vojkh/GSE140203_3T3_HG19_atac2gex.zip?rlkey=xmxrwso5e5ty3w53ctbm5bo9z&dl=1",
        "GSE140203_3T3_MM10_atac2gex":
        "https://www.dropbox.com/scl/fi/po9k064twny51subze6df/GSE140203_3T3_MM10_atac2gex.zip?rlkey=q0b4y58bsvacnjrmvsclk4jqu&dl=1",
        "GSE140203_12878.rep2_atac2gex":
        "https://www.dropbox.com/scl/fi/jqijimb7h6cv4w4hkax1q/GSE140203_12878.rep2_atac2gex.zip?rlkey=c837xkoacap4wjszffpfrmuak&dl=1",
        "GSE140203_12878.rep3_atac2gex":
        "https://www.dropbox.com/scl/fi/wlv9dhvylz78kq8ezncmd/GSE140203_12878.rep3_atac2gex.zip?rlkey=5r607plnqzlxdgxtc4le8d6o1&dl=1",
        "GSE140203_K562_HG19_atac2gex":
        "https://www.dropbox.com/scl/fi/n2he1br3u604p3mgniowz/GSE140203_K562_HG19_atac2gex.zip?rlkey=2lhe7s5run8ly5uk4b0vfemyj&dl=1",
        "GSE140203_K562_MM10_atac2gex":
        "https://www.dropbox.com/scl/fi/dhdorqy87915uah3xl07a/GSE140203_K562_MM10_atac2gex.zip?rlkey=ecwsy5sp7f2i2gtjo1qyaf4zt&dl=1",
        "GSE140203_LUNG_atac2gex":
        "https://www.dropbox.com/scl/fi/gabugiw244ky85j3ckq4d/GSE140203_LUNG_atac2gex.zip?rlkey=uj0we276s6ay2acpioj4tmfj3&dl=1"
    }
    SUBTASK_NAME_MAP = {
        "adt2gex": "openproblems_bmmc_cite_phase2_mod2",
        "atac2gex": "openproblems_bmmc_multiome_phase2_mod2",
        "gex2adt": "openproblems_bmmc_cite_phase2_rna",
        "gex2atac": "openproblems_bmmc_multiome_phase2_rna",
        "gex2adt_subset": "openproblems_bmmc_cite_phase2_rna_subset",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data", preprocess=None, span=0.3):
        # TODO: factor our preprocess
        self.preprocess = preprocess
        self.span = span
        super().__init__(subtask, root)

    def _raw_to_dance(self, raw_data):
        train_mod1, train_mod2, test_mod1, test_mod2 = self._maybe_preprocess(raw_data)

        mod1 = ad.concat((train_mod1, test_mod1))
        mod2 = ad.concat((train_mod2, test_mod2))
        mod1.var_names_make_unique()
        mod2.var_names_make_unique()

        mdata = md.MuData({"mod1": mod1, "mod2": mod2})
        mdata.var_names_make_unique()

        data = Data(mdata, train_size=train_mod1.shape[0])
        data.set_config(feature_mod="mod1", label_mod="mod2")

        return data

    def _maybe_preprocess(self, raw_data, selection_threshold=10000):
        if self.preprocess == "feature_selection":
            if raw_data[0].shape[1] > selection_threshold:
                sc.pp.highly_variable_genes(raw_data[0], layer="counts", flavor="seurat_v3",
                                            n_top_genes=selection_threshold, span=self.span)
                raw_data[2].var["highly_variable"] = raw_data[0].var["highly_variable"]
                for i in [0, 2]:
                    raw_data[i] = raw_data[i][:, raw_data[i].var["highly_variable"]]
        elif self.preprocess not in (None, "none"):
            logger.info(f"Preprocessing method {self.preprocess!r} not supported.")
        logger.info("Preprocessing done.")
        return raw_data


class ModalityMatchingDataset(MultiModalityDataset):

    TASK = "match_modality"
    URL_DICT = {
        "openproblems_bmmc_cite_phase2_mod2":
        "https://www.dropbox.com/s/fa6zut89xx73itz/openproblems_bmmc_cite_phase2_mod2.zip?dl=1",
        "openproblems_bmmc_cite_phase2_rna":
        "https://www.dropbox.com/s/ep00mqcjmdu0b7v/openproblems_bmmc_cite_phase2_rna.zip?dl=1",
        "openproblems_bmmc_multiome_phase2_mod2":
        "https://www.dropbox.com/s/31qi5sckx768acw/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1",
        "openproblems_bmmc_multiome_phase2_rna":
        "https://www.dropbox.com/s/h1s067wkefs1jh2/openproblems_bmmc_multiome_phase2_rna.zip?dl=1",
        "openproblems_bmmc_cite_phase2_rna_subset":
        "https://www.dropbox.com/s/3q4xwpzjbe81x58/openproblems_bmmc_cite_phase2_rna_subset.zip?dl=1",
    }
    SUBTASK_NAME_MAP = {
        "adt2gex": "openproblems_bmmc_cite_phase2_mod2",
        "atac2gex": "openproblems_bmmc_multiome_phase2_mod2",
        "gex2adt": "openproblems_bmmc_cite_phase2_rna",
        "gex2atac": "openproblems_bmmc_multiome_phase2_rna",
        "gex2adt_subset": "openproblems_bmmc_cite_phase2_rna_subset",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data", preprocess=None, pkl_path=None, span=0.3):
        # TODO: factor our preprocess
        self.preprocess = preprocess
        self.pkl_path = pkl_path
        self.span = span
        super().__init__(subtask, root)

    def _raw_to_dance(self, raw_data):
        train_mod1, train_mod2, train_label, test_mod1, test_mod2, test_label = self._maybe_preprocess(raw_data)
        # Align matched cells
        train_mod2 = train_mod2[train_label.to_df().values.argmax(1)]

        mod1 = ad.concat((train_mod1, test_mod1))
        mod2 = ad.concat((train_mod2, test_mod2))
        mod1.var_names_make_unique()
        mod2.var_names_make_unique()
        mod2.obs_names = mod1.obs_names
        train_size = train_mod1.shape[0]

        mod1.obsm["labels"] = np.concatenate([np.zeros(train_size), np.argmax(test_label.X.toarray(), 1)])

        # Combine modalities into mudata
        mdata = md.MuData({"mod1": mod1, "mod2": mod2})
        mdata.var_names_make_unique()

        data = Data(mdata, train_size=train_size)

        return data

    def _maybe_preprocess(self, raw_data, selection_threshold=10000):
        if self.preprocess is None:
            return raw_data

        train_mod1, train_mod2, train_label, test_mod1, test_mod2, test_label = raw_data
        modalities = [train_mod1, train_mod2, test_mod1, test_mod2]

        # TODO: support other two subtasks
        assert self.subtask in ("openproblems_bmmc_cite_phase2_rna", "openproblems_bmmc_cite_phase2_rna_subset",
                                "openproblems_bmmc_multiome_phase2_rna"), "Currently not available."

        if self.preprocess == "pca":
            if self.pkl_path and osp.exists(self.pkl_path):
                with open(self.pkl_path, "rb") as f:
                    preprocessed_features = pickle.load(f)

            else:
                if self.subtask in ("openproblems_bmmc_cite_phase2_rna", "openproblems_bmmc_cite_phase2_rna_subset"):
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(modalities[2]).values
                    m2_train = modalities[1].X.toarray()
                    m2_test = modalities[3].X.toarray()

                elif self.subtask == "openproblems_bmmc_multiome_phase2_rna":
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(modalities[2]).values
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    m2_train = lsi_transformer_atac.fit_transform(modalities[1]).values
                    m2_test = lsi_transformer_atac.transform(modalities[3]).values

                else:
                    raise ValueError(f"Unrecognized subtask name: {self.subtask}")

                preprocessed_features = {
                    "mod1_train": m1_train,
                    "mod2_train": m2_train,
                    "mod1_test": m1_test,
                    "mod2_test": m2_test
                }

                if self.pkl_path:
                    with open(self.pkl_path, "wb") as f:
                        pickle.dump(preprocessed_features, f)

            modalities[0].obsm["X_pca"] = preprocessed_features["mod1_train"]
            modalities[1].obsm["X_pca"] = preprocessed_features["mod2_train"]
            modalities[2].obsm["X_pca"] = preprocessed_features["mod1_test"]
            modalities[3].obsm["X_pca"] = preprocessed_features["mod2_test"]

        elif self.preprocess == "feature_selection":
            for i in range(2):
                if modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(modalities[i], layer="counts", flavor="seurat_v3",
                                                n_top_genes=selection_threshold, span=self.span)
                    modalities[i + 2].var["highly_variable"] = modalities[i].var["highly_variable"]
                    modalities[i] = modalities[i][:, modalities[i].var["highly_variable"]]
                    modalities[i + 2] = modalities[i + 2][:, modalities[i + 2].var["highly_variable"]]

        else:
            logger.info("Preprocessing method not supported.")

        logger.info("Preprocessing done.")

        train_mod1, train_mod2, test_mod1, test_mod2 = modalities
        return train_mod1, train_mod2, train_label, test_mod1, test_mod2, test_label


class JointEmbeddingNIPSDataset(MultiModalityDataset):

    TASK = "joint_embedding"
    URL_DICT = {
        "openproblems_bmmc_cite_phase2":
        "https://www.dropbox.com/s/hjr4dxuw55vin5z/openproblems_bmmc_cite_phase2.zip?dl=1",
        "openproblems_bmmc_multiome_phase2":
        "https://www.dropbox.com/s/40kjslupxhkg92s/openproblems_bmmc_multiome_phase2.zip?dl=1"
    }
    SUBTASK_NAME_MAP = {
        "adt": "openproblems_bmmc_cite_phase2",
        "atac": "openproblems_bmmc_multiome_phase2",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data", preprocess=None, normalize=False, pretrained_folder="."):
        # TODO: factor our preprocess
        self.preprocess = preprocess
        self.normalize = normalize
        self.pretrained_folder = pretrained_folder
        super().__init__(subtask, root)

    def _raw_to_dance(self, raw_data):
        mod1, mod2, meta1, meta2, test_sol = self._maybe_preprocess(raw_data)

        assert all(mod2.obs_names == mod1.obs_names), "Modalities not aligned"
        mdata = md.MuData({"mod1": mod1, "mod2": mod2, "meta1": meta1, "meta2": meta2, "test_sol": test_sol})

        train_size = meta1.shape[0]
        data = Data(mdata, train_size=train_size)

        return data

    def _maybe_preprocess(self, raw_data, selection_threshold=10000):
        if self.preprocess is None:
            return raw_data

        mod1, mod2, meta1, meta2, test_sol = raw_data
        train_size = meta1.shape[0]

        # aux -> cell cycle analysis
        if self.preprocess == "aux":
            os.makedirs(self.pretrained_folder, exist_ok=True)

            if osp.exists(osp.join(self.pretrained_folder, f"preprocessed_data_{self.subtask}.pkl")):

                with open(osp.join(self.pretrained_folder, f"preprocessed_data_{self.subtask}.pkl"), "rb") as f:
                    preprocessed_data = pickle.load(f)

                    y_train = preprocessed_data["y_train"]
                    mod1.obsm["X_pca"] = preprocessed_data["X_pca_0"]
                    mod2.obsm["X_pca"] = preprocessed_data["X_pca_1"]

                    mod1.obsm["cell_type"] = y_train[0]
                    mod1.obsm["batch_label"] = np.concatenate(
                        [y_train[1], np.zeros(y_train[0].shape[0] - train_size)], 0)
                    mod1.obsm["phase_labels"] = np.concatenate(
                        [y_train[2], np.zeros(y_train[0].shape[0] - train_size)], 0)
                    mod1.obsm["S_scores"] = np.concatenate([y_train[3], np.zeros(y_train[0].shape[0] - train_size)], 0)
                    mod1.obsm["G2M_scores"] = np.concatenate(
                        [y_train[4], np.zeros(y_train[0].shape[0] - train_size)], 0)

                with open(osp.join(self.pretrained_folder, f"{self.subtask}_config.pk"), "rb") as f:
                    # cell types, batch labels, cell cycle
                    self.nb_cell_types, self.nb_batches, self.nb_phases = pickle.load(f)
                logger.info("Preprocessing done.")
                return mod1, mod2, meta1, meta2, test_sol

            # PCA
            mod1_name = mod1.var["feature_types"][0]
            mod2_name = mod2.var["feature_types"][0]
            if mod2_name == "ADT":
                if osp.exists(osp.join(self.pretrained_folder, f"lsi_cite_{mod1_name}.pkl")):
                    with open(osp.join(self.pretrained_folder, f"lsi_cite_{mod1_name}.pkl"), "rb") as f:
                        lsi_transformer_gex = pickle.load(f)
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    lsi_transformer_gex.fit(mod1)
                    with open(osp.join(self.pretrained_folder, f"lsi_cite_{mod1_name}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_gex, f)

            if mod2_name == "ATAC":

                if osp.exists(osp.join(self.pretrained_folder, f"lsi_multiome_{mod1_name}.pkl")):
                    with open(osp.join(self.pretrained_folder, f"lsi_multiome_{mod1_name}.pkl"), "rb") as f:
                        lsi_transformer_gex = pickle.load(f)
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=64, drop_first=True)
                    lsi_transformer_gex.fit(mod1)
                    with open(osp.join(self.pretrained_folder, f"lsi_multiome_{mod1_name}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_gex, f)

                if osp.exists(osp.join(self.pretrained_folder, f"lsi_multiome_{mod2_name}.pkl")):
                    with open(osp.join(self.pretrained_folder, f"lsi_multiome_{mod2_name}.pkl"), "rb") as f:
                        lsi_transformer_atac = pickle.load(f)
                else:
                    #         lsi_transformer_atac = TruncatedSVD(n_components=100, random_state=random_seed)
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    lsi_transformer_atac.fit(mod2)
                    with open(osp.join(self.pretrained_folder, f"lsi_multiome_{mod2_name}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_atac, f)

            # Data preprocessing
            # Only exploration dataset provides cell type information.
            # The exploration dataset is a subset of the full dataset.
            ad_mod1 = meta1
            mod1_obs = ad_mod1.obs

            # Make sure exploration data match the full data
            assert ((mod1.obs["batch"].index[:mod1_obs.shape[0]] == mod1_obs["batch"].index).mean() == 1)

            if mod2_name == "ADT":
                mod1_pca = lsi_transformer_gex.transform(mod1).values
                mod2_pca = mod2.X.toarray()
            elif mod2_name == "ATAC":
                mod1_pca = lsi_transformer_gex.transform(mod1).values
                mod2_pca = lsi_transformer_atac.transform(mod2).values
            else:
                raise ValueError(f"Unknown modality 2: {mod2_name}")

            cell_cycle_genes = [
                "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1",
                "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
                "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
                "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1", "UBE2C",
                "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
                "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B",
                "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2",
                "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2",
                "G2E3", "GAS2L3", "CBX5", "CENPA"
            ]
            logger.info(f"Data loading and pca done: {mod1_pca.shape=}, {mod2_pca.shape=}")
            logger.info("Start to calculate cell_cycle score. It may roughly take an hour.")

            cell_type_labels = test_sol.obs["cell_type"].to_numpy()
            batch_ids = mod1_obs["batch"]
            phase_labels = mod1_obs["phase"]
            nb_cell_types = len(np.unique(cell_type_labels))
            nb_batches = len(np.unique(batch_ids))
            nb_phases = len(np.unique(phase_labels)) - 1  # 2
            cell_type_labels_unique = list(np.unique(cell_type_labels))
            batch_ids_unique = list(np.unique(batch_ids))
            phase_labels_unique = list(np.unique(phase_labels))
            c_labels = np.array([cell_type_labels_unique.index(item) for item in cell_type_labels])
            b_labels = np.array([batch_ids_unique.index(item) for item in batch_ids])
            p_labels = np.array([phase_labels_unique.index(item) for item in phase_labels])
            # 0:G1, 1:G2M, 2: S, only consider the last two
            s_genes = cell_cycle_genes[:43]
            g2m_genes = cell_cycle_genes[43:]
            sc.pp.log1p(ad_mod1)
            sc.pp.scale(ad_mod1)
            sc.tl.score_genes_cell_cycle(ad_mod1, s_genes=s_genes, g2m_genes=g2m_genes)
            S_scores = ad_mod1.obs["S_score"].values
            G2M_scores = ad_mod1.obs["G2M_score"].values
            # phase_scores = np.stack([S_scores, G2M_scores]).T  # (nb_cells, 2)

            y_train = [c_labels, b_labels, p_labels, S_scores, G2M_scores]
            mod1.obsm["X_pca"] = mod1_pca
            mod2.obsm["X_pca"] = mod2_pca
            train_size = mod1_obs.shape[0]
            mod1.obsm["cell_type"] = c_labels
            mod1.obsm["batch_label"] = np.concatenate([y_train[1], np.zeros(mod1.shape[0] - train_size)], 0)
            mod1.obsm["phase_labels"] = np.concatenate([y_train[2], np.zeros(mod1.shape[0] - train_size)], 0)
            mod1.obsm["S_scores"] = np.concatenate([y_train[3], np.zeros(mod1.shape[0] - train_size)], 0)
            mod1.obsm["G2M_scores"] = np.concatenate([y_train[4], np.zeros(mod1.shape[0] - train_size)], 0)

            preprocessed_data = {"X_pca_0": mod1.obsm["X_pca"], "X_pca_1": mod2.obsm["X_pca"], "y_train": y_train}

            with open(osp.join(self.pretrained_folder, f"preprocessed_data_{self.subtask}.pkl"), "wb") as f:
                pickle.dump(preprocessed_data, f)

            with open(osp.join(self.pretrained_folder, f"{self.subtask}_config.pk"), "wb") as f:
                pickle.dump([nb_cell_types, nb_batches, nb_phases], f)

            self.nb_cell_types, self.nb_batches, self.nb_phases = nb_cell_types, nb_batches, nb_phases

        elif self.preprocess == "feature_selection":
            if mod1.shape[1] > selection_threshold:
                sc.pp.highly_variable_genes(mod1, layer="counts", flavor="seurat_v3", n_top_genes=selection_threshold)
                mod1 = mod1[:, mod1.var["highly_variable"]]

            if mod2.shape[1] > selection_threshold:
                sc.pp.highly_variable_genes(mod2, layer="counts", flavor="seurat_v3", n_top_genes=selection_threshold)
                mod2 = mod2[:, mod2.var["highly_variable"]]

        else:
            logger.info(f"Preprocessing method {self.preprocess!r} not supported.")

        # Normalization
        if self.normalize:
            sc.pp.scale(mod1)
            sc.pp.scale(mod2)

        logger.info("Preprocessing done.")

        return mod1, mod2, meta1, meta2, test_sol
