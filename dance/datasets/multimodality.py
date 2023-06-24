import os
import os.path as osp
import pickle
from abc import ABC

import anndata as ad
import mudata as md
import numpy as np
import scanpy as sc
import torch

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
        self.data_url = self.URL_DICT[subtask]
        self.loaded = False

        super().__init__(root=root, full_download=False)

    def download(self):
        self.download_data()

    def download_data(self):
        download_file(self.data_url, osp.join(self.root, f"{self.subtask}.zip"))
        unzip_file(osp.join(self.root, f"{self.subtask}.zip"), self.root)
        return self

    def download_pathway(self):
        download_file("https://www.dropbox.com/s/uqoakpalr3albiq/h.all.v7.4.entrez.gmt?dl=1",
                      osp.join(self.root, "h.all.v7.4.entrez.gmt"))
        download_file("https://www.dropbox.com/s/yjrcsd2rpmahmfo/h.all.v7.4.symbols.gmt?dl=1",
                      osp.join(self.root, "h.all.v7.4.symbols.gmt"))
        return self

    @property
    def mod_data_paths(self) -> List[str]:
        if self.TASK == "joint_embedding":
            paths = [
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_mod2.h5ad"),
            ]
        else:
            paths = [
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_mod2.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod1.h5ad"),
                osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_mod2.h5ad"),
            ]
        return paths

    def is_complete(self) -> bool:
        return all(map(osp.exists, self.mod_data_paths))

    def _load_raw_data(self) -> List[ad.AnnData]:
        modalities = []
        for mod_path in self.mod_data_paths:
            logger.info(f"Loading {mod_path}")
            modalities.append(ad.read_h5ad(mod_path))
        self.loaded = True
        return modalities

    def sparse_features(self, index=None, count=False):
        assert self.loaded, "Data have not been loaded."
        if not count:
            if index is None:
                return [mod.X for mod in self.modalities]
            else:
                return self.modalities[index].X
        else:
            if index is None:
                return [mod.layers["counts"] for mod in self.modalities]
            else:
                return self.modalities[index].layers["counts"]

    def numpy_features(self, index=None, count=False):
        assert self.loaded, "Data have not been loaded."
        if not count:
            if index is None:
                return [mod.X.toarray() for mod in self.modalities]
            else:
                return self.modalities[index].X.toarray()
        else:
            if index is None:
                return [mod.layers["counts"].toarray() for mod in self.modalities]
            else:
                return self.modalities[index].layers["counts"].toarray()

    def tensor_features(self, index=None, count=False, device="cpu"):
        assert self.loaded, "Data have not been loaded."
        if not count:
            if index is None:
                return [torch.from_numpy(mod.X.toarray()).to(device) for mod in self.modalities]
            else:
                return torch.from_numpy(self.modalities[index].X.toarray()).to(device)
        else:
            if index is None:
                return [torch.from_numpy(mod.layers["counts"].toarray()).to(device) for mod in self.modalities]
            else:
                return torch.from_numpy(self.modalities[index].layers["counts"].toarray()).to(device)

    def get_modalities(self):
        assert self.loaded, "Data have not been loaded."
        return self.modalities


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
    }
    SUBTASK_NAME_MAP = {
        "adt2gex": "openproblems_bmmc_cite_phase2_mod2",
        "atac2gex": "openproblems_bmmc_multiome_phase2_mod2",
        "gex2adt": "openproblems_bmmc_cite_phase2_rna",
        "gex2atac": "openproblems_bmmc_multiome_phase2_rna",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data", preprocess=None):
        # TODO: factor our preprocess
        self.preprocess = preprocess
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
                                            n_top_genes=selection_threshold)
                raw_data[2].var["highly_variable"] = raw_data[0].var["highly_variable"]
                for i in [0, 2]:
                    raw_data[i] = raw_data[i][:, raw_data[i].var["highly_variable"]]
        elif self.preprocess is not None:
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
        "https://www.dropbox.com/s/h1s067wkefs1jh2/openproblems_bmmc_multiome_phase2_rna.zip?dl=1"
    }
    SUBTASK_NAME_MAP = {
        "adt2gex": "openproblems_bmmc_cite_phase2_mod2",
        "atac2gex": "openproblems_bmmc_multiome_phase2_mod2",
        "gex2adt": "openproblems_bmmc_cite_phase2_rna",
        "gex2atac": "openproblems_bmmc_multiome_phase2_rna",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data"):
        super().__init__(subtask, root)
        self.preprocessed = False

    def load_sol(self):
        assert (self.loaded)
        self.train_sol = ad.read_h5ad(
            osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_train_sol.h5ad"))
        self.test_sol = ad.read_h5ad(
            osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_test_sol.h5ad"))
        self.modalities[1] = self.modalities[1][self.train_sol.to_df().values.argmax(1)]
        return self

    def preprocess(self, kind="pca", pkl_path=None, selection_threshold=10000):

        # TODO: support other two subtasks
        assert self.subtask in ("openproblems_bmmc_cite_phase2_rna",
                                "openproblems_bmmc_multiome_phase2_rna"), "Currently not available."

        if kind == "pca":
            if pkl_path and (not osp.exists(pkl_path)):

                if self.subtask == "openproblems_bmmc_cite_phase2_rna":
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                    m2_train = self.modalities[1].X.toarray()
                    m2_test = self.modalities[3].X.toarray()

                elif self.subtask == "openproblems_bmmc_multiome_phase2_rna":
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    m2_train = lsi_transformer_atac.fit_transform(self.modalities[1]).values
                    m2_test = lsi_transformer_atac.transform(self.modalities[3]).values

                else:
                    raise ValueError(f"Unrecognized subtask name: {self.subtask}")

                self.preprocessed_features = {
                    "mod1_train": m1_train,
                    "mod2_train": m2_train,
                    "mod1_test": m1_test,
                    "mod2_test": m2_test
                }
                self.modalities[0].obsm["X_pca"] = self.preprocessed_features["mod1_train"]
                self.modalities[1].obsm["X_pca"] = self.preprocessed_features["mod2_train"]
                self.modalities[2].obsm["X_pca"] = self.preprocessed_features["mod1_test"]
                self.modalities[3].obsm["X_pca"] = self.preprocessed_features["mod2_test"]
                pickle.dump(self.preprocessed_features, open(pkl_path, "wb"))

            else:
                self.preprocessed_features = pickle.load(open(pkl_path, "rb"))
                self.modalities[0].obsm["X_pca"] = self.preprocessed_features["mod1_train"]
                self.modalities[1].obsm["X_pca"] = self.preprocessed_features["mod2_train"]
                self.modalities[2].obsm["X_pca"] = self.preprocessed_features["mod1_test"]
                self.modalities[3].obsm["X_pca"] = self.preprocessed_features["mod2_test"]
        elif kind == "feature_selection":
            for i in range(2):
                if self.modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(self.modalities[i], layer="counts", flavor="seurat_v3",
                                                n_top_genes=selection_threshold)
                    self.modalities[i + 2].var["highly_variable"] = self.modalities[i].var["highly_variable"]
                    self.modalities[i] = self.modalities[i][:, self.modalities[i].var["highly_variable"]]
                    self.modalities[i + 2] = self.modalities[i + 2][:, self.modalities[i + 2].var["highly_variable"]]
        else:
            logger.info("Preprocessing method not supported.")
            return self
        logger.info("Preprocessing done.")
        self.preprocessed = True
        return self

    def get_preprocessed_features(self):
        assert self.preprocessed, "Transformed features do not exist."
        return self.preprocessed_features


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

    def __init__(self, subtask, root="./data"):
        super().__init__(subtask, root)
        self.preprocessed = False

    def load_metadata(self):
        assert (self.loaded)

        if self.subtask.find("cite") != -1:
            mod = "adt"
            meta = "cite"
        else:
            mod = "atac"
            meta = "multiome"
        self.exploration = [
            ad.read_h5ad(osp.join(self.root, self.subtask, f"{meta}_gex_processed_training.h5ad")),
            ad.read_h5ad(osp.join(self.root, self.subtask, f"{meta}_{mod}_processed_training.h5ad")),
        ]
        return self

    def load_sol(self):
        assert (self.loaded)
        self.test_sol = ad.read_h5ad(
            osp.join(self.root, self.subtask, f"{self.subtask}.censor_dataset.output_solution.h5ad"))
        return self

    def preprocess(self, kind="aux", pretrained_folder=".", selection_threshold=10000):
        # aux -> cell cycle analysis
        if kind == "aux":
            os.makedirs(pretrained_folder, exist_ok=True)

            if osp.exists(osp.join(pretrained_folder, f"preprocessed_data_{self.subtask}.pkl")):

                with open(osp.join(pretrained_folder, f"preprocessed_data_{self.subtask}.pkl"), "rb") as f:
                    self.preprocessed_data = pickle.load(f)
                    Y_train = self.preprocessed_data["Y_train"]
                    self.modalities[0].obsm["X_pca"] = self.preprocessed_data["X_pca_0"]
                    self.modalities[1].obsm["X_pca"] = self.preprocessed_data["X_pca_1"]
                    self.train_size = self.exploration[0].shape[0]
                    self.modalities[0].obsm["cell_type"] = Y_train[0]
                    self.modalities[0].obsm["batch_label"] = np.concatenate(
                        [Y_train[1], np.zeros(Y_train[0].shape[0] - self.train_size)], 0)
                    self.modalities[0].obsm["phase_labels"] = np.concatenate(
                        [Y_train[2], np.zeros(Y_train[0].shape[0] - self.train_size)], 0)
                    self.modalities[0].obsm["S_scores"] = np.concatenate(
                        [Y_train[3], np.zeros(Y_train[0].shape[0] - self.train_size)], 0)
                    self.modalities[0].obsm["G2M_scores"] = np.concatenate(
                        [Y_train[4], np.zeros(Y_train[0].shape[0] - self.train_size)], 0)

                with open(osp.join(pretrained_folder, f"{self.subtask}_config.pk"), "rb") as f:
                    # cell types, batch labels, cell cycle
                    self.nb_cell_types, self.nb_batches, self.nb_phases = pickle.load(f)
                self.preprocessed = True
                logger.info("Preprocessing done.")
                return self

            ##########################################
            ##             PCA PRETRAIN             ##
            ##########################################

            # scale and log transform

            mod1 = self.modalities[0].var["feature_types"][0]
            mod2 = self.modalities[1].var["feature_types"][0]

            if mod2 == "ADT":
                if osp.exists(osp.join(pretrained_folder, f"lsi_cite_{mod1}.pkl")):
                    lsi_transformer_gex = pickle.load(open(osp.join(pretrained_folder, f"lsi_cite_{mod1}.pkl"), "rb"))
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    lsi_transformer_gex.fit(self.modalities[0])
                    pickle.dump(lsi_transformer_gex, open(osp.join(pretrained_folder, f"lsi_cite_{mod1}.pkl"), "wb"))

            if mod2 == "ATAC":

                if osp.exists(osp.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl")):
                    with open(osp.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl"), "rb") as f:
                        lsi_transformer_gex = pickle.load(f)
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=64, drop_first=True)
                    lsi_transformer_gex.fit(self.modalities[0])
                    with open(osp.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_gex, f)

                if osp.exists(osp.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl")):
                    with open(osp.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl"), "rb") as f:
                        lsi_transformer_atac = pickle.load(f)
                else:
                    #         lsi_transformer_atac = TruncatedSVD(n_components=100, random_state=random_seed)
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    lsi_transformer_atac.fit(self.modalities[1])
                    with open(osp.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_atac, f)

            ##########################################
            ##           DATA PREPROCESSING         ##
            ##########################################

            # Only exploration dataset provides cell type information.
            # The exploration dataset is a subset of the full dataset.
            ad_mod1 = self.exploration[0]
            mod1_obs = ad_mod1.obs

            # Make sure exploration data match the full data
            assert ((self.modalities[0].obs["batch"].index[:mod1_obs.shape[0]] == mod1_obs["batch"].index).mean() == 1)

            if mod2 == "ADT":
                # mod1_pca = lsi_transformer_gex.transform(ad_mod1).values
                # mod1_pca_test = lsi_transformer_gex.transform(self.modalities[0][mod1_obs.shape[0]:]).values
                # mod2_pca = ad_mod2.X.toarray()
                # mod2_pca_test = self.numpy_features(1)[mod1_obs.shape[0]:]

                mod1_pca = lsi_transformer_gex.transform(self.modalities[0]).values
                mod2_pca = self.numpy_features(1)
            elif mod2 == "ATAC":
                # mod1_pca = lsi_transformer_gex.transform(ad_mod1).values
                # mod1_pca_test = lsi_transformer_gex.transform(self.modalities[0][mod1_obs.shape[0]:]).values
                # mod2_pca = lsi_transformer_atac.transform(ad_mod2).values
                # mod2_pca_test = lsi_transformer_atac.transform(self.modalities[1][mod1_obs.shape[0]:]).values

                mod1_pca = lsi_transformer_gex.transform(self.modalities[0]).values
                mod2_pca = lsi_transformer_atac.transform(self.modalities[1]).values
            else:
                raise ValueError(f"Unknown modality 2: {mod2}")

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

            logger.info("Data loading and pca done", mod1_pca.shape, mod2_pca.shape)
            logger.info("Start to calculate cell_cycle score. It may roughly take an hour.")

            cell_type_labels = self.test_sol.obs["cell_type"].to_numpy()
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

            Y_train = [c_labels, b_labels, p_labels, S_scores, G2M_scores]
            self.modalities[0].obsm["X_pca"] = mod1_pca
            self.modalities[1].obsm["X_pca"] = mod2_pca
            self.train_size = mod1_obs.shape[0]
            self.modalities[0].obsm["cell_type"] = cell_type_labels
            self.modalities[0].obsm["batch_label"] = np.concatenate(
                [Y_train[1], np.zeros(self.modalities[0].shape[0] - self.train_size)], 0)
            self.modalities[0].obsm["phase_labels"] = np.concatenate(
                [Y_train[2], np.zeros(self.modalities[0].shape[0] - self.train_size)], 0)
            self.modalities[0].obsm["S_scores"] = np.concatenate(
                [Y_train[3], np.zeros(self.modalities[0].shape[0] - self.train_size)], 0)
            self.modalities[0].obsm["G2M_scores"] = np.concatenate(
                [Y_train[4], np.zeros(self.modalities[0].shape[0] - self.train_size)], 0)

            self.preprocessed_data = {
                "X_pca_0": self.modalities[0].obsm["X_pca"],
                "X_pca_1": self.modalities[1].obsm["X_pca"],
                "Y_train": Y_train
            }
            pickle.dump(self.preprocessed_data,
                        open(osp.join(pretrained_folder, f"preprocessed_data_{self.subtask}.pkl"), "wb"))
            pickle.dump([nb_cell_types, nb_batches, nb_phases],
                        open(osp.join(pretrained_folder, f"{self.subtask}_config.pk"), "wb"))

            self.nb_cell_types, self.nb_batches, self.nb_phases = nb_cell_types, nb_batches, nb_phases
        elif kind == "feature_selection":
            for i in range(2):
                if self.modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(self.modalities[i], layer="counts", flavor="seurat_v3",
                                                n_top_genes=selection_threshold)
                    self.modalities[i] = self.modalities[i][:, self.modalities[i].var["highly_variable"]]
        else:
            logger.info("Preprocessing method not supported.")
            return self
        self.preprocessed = True
        logger.info("Preprocessing done.")
        return self

    def get_preprocessed_data(self):
        return self.preprocessed_data

    def normalize(self):
        assert self.preprocessed, "Normalization must be conducted after preprocessing."

        self.mean0 = self.modalities[0].obsm["X_pca"].mean()
        self.mean1 = self.modalities[1].obsm["X_pca"].mean()
        self.std0 = self.modalities[0].obsm["X_pca"].std()
        self.std1 = self.modalities[1].obsm["X_pca"].std()
        self.modalities[0].obsm["X_pca"] = (self.modalities[0].obsm["X_pca"] - self.mean0) / self.std0
        self.modalities[1].obsm["X_pca"] = (self.modalities[1].obsm["X_pca"] - self.mean1) / self.std1

        return self
