import os
import os.path as osp
import pickle
from abc import ABC

import anndata as ad
import mudata as md
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from dance import logger
from dance.data import Data
from dance.datasets.base import BaseDataset
from dance.registry import register_dataset
from dance.transforms.preprocess import lsiTransformer
from dance.typing import List
from dance.utils import is_numeric
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
            if self.subtask.startswith("GSE140203"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_solution.h5ad"),
                ]
            if self.subtask.startswith("openproblems_2022"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_solution.h5ad"),
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
            if self.subtask == "pbmc_cite":
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.citeanti_dataset.output_test_sol.h5ad"),
                ]
            if self.subtask.startswith("openproblems_2022"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.open_dataset.output_test_sol.h5ad"),
                ]
            if self.subtask.startswith("GSE127064"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE126074_dataset.output_test_sol.h5ad")
                ]
            if self.subtask.startswith("GSE117089"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE117089_dataset.output_test_sol.h5ad")
                ]
            if self.subtask.startswith("GSE140203"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.GSE140203_dataset.output_test_sol.h5ad"),
                ]
            if self.subtask == "10k_pbmc":
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.10kanti_dataset_subset.output_test_sol.h5ad")
                ]
            if self.subtask.startswith("5k_pbmc"):
                paths = [
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_train_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_train_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_train_sol.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_test_mod1.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_test_mod2.h5ad"),
                    osp.join(self.root, self.subtask, f"{self.subtask}.5kanti_dataset.output_test_sol.h5ad"),
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


@register_dataset("multimodality")
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
        "https://www.dropbox.com/scl/fi/luzmc1jab7zvvxi2i4od5/openproblems_2022_multi_atac2gex.zip?rlkey=ht1acmhdpq8bbo1guevqgej5y&dl=1",
        "openproblems_2022_cite_gex2adt":
        "https://www.dropbox.com/scl/fi/ejioe3qqug0h2f7wvw9hq/openproblems_2022_cite_gex2adt.zip?rlkey=2f9kqz61s9ixdllgzic9tamc7&dl=1",
        "GSE127064_AdBrain_gex2atac":
        "https://www.dropbox.com/scl/fi/4ybsx6pgiuy6j9m0y92ly/GSE127064_AdBrain_gex2atac.zip?rlkey=6a5u7p7xr2dqsoduflzxjluja&dl=1",
        "GSE127064_p0Brain_gex2atac":
        "https://www.dropbox.com/scl/fi/k4p3nkkqq56ev6ljyo5se/GSE127064_p0Brain_gex2atac.zip?rlkey=y7kayqmk2l72jjogzlvfxtl74&dl=1",
        "GSE117089_mouse_gex2atac":
        "https://www.dropbox.com/scl/fi/hbo5eel8vtkctwhgelu5u/GSE117089_mouse_gex2atac.zip?rlkey=84t4kj1ls7ut09dpcbj86mtlc&dl=1",
        "GSE117089_sciCAR_gex2atac":
        "https://www.dropbox.com/scl/fi/hc0c48so824uohx0szs3h/GSE117089_sciCAR_gex2atac.zip?rlkey=4xjayirgijodd1fqcf7a42apo&dl=1",
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

        changed_count = 0  # keep track to modified entries due to ensuring count data type
        for i in range(4):
            m_data = raw_data[i].X
            int_data = m_data.astype(int)
            changed_count += np.sum(int_data != m_data)
            raw_data[i].X = int_data
            raw_data[i].layers["counts"] = raw_data[i].X
        if changed_count > 0:
            logger.warning("Implicit modification: to ensure count (integer type) data, "
                           f"a total number of {changed_count} entries were modified.")

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


@register_dataset("multimodality")
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
        "pbmc_cite":
        "https://www.dropbox.com/scl/fi/eq9eg6hzoqj2plgi2003k/pbmc_cite.zip?rlkey=p7bgttr7v99jxu3qem7sh8qrh&dl=1",
        "openproblems_2022_multi_atac2gex_subset":
        "https://www.dropbox.com/scl/fi/2p8izdu5xwvgm705hdf16/openproblems_2022_multi_atac2gex_subset.zip?rlkey=v962rncxmc9jqab2vhk3438sp&dl=1",
        "openproblems_2022_cite_gex2adt_subset":
        "https://www.dropbox.com/scl/fi/o9ht00cqgkxwgixtaydxm/openproblems_2022_cite_gex2adt_subset.zip?rlkey=sqnodvi25btk1igowww2pen8h&dl=1",
        "5k_pbmc_subset":
        "https://www.dropbox.com/scl/fi/rhyzaqtxpkvcu2za8mqaq/5k_pbmc_subset.zip?rlkey=g019vyku5let92z814dor287w&dl=1",
        "10k_pbmc":
        "https://www.dropbox.com/scl/fi/1wi9u5zwzx7td9akk1cri/10k_pbmc.zip?rlkey=u9ir7b6d8s3t29sk2hu7v29au&dl=1",
        "GSE117089_mouse_gex2atac":
        "https://www.dropbox.com/scl/fi/dbxgretuwq1zekxibb2p0/GSE117089_mouse_gex2atac.zip?rlkey=wzqi309on9v1wllkiatnkpnhv&dl=1",
        "GSE117089_sciCAR_gex2atac":
        "https://www.dropbox.com/scl/fi/4sohkymkqyry5xkx34oiw/GSE117089_sciCAR_gex2atac.zip?rlkey=6exg6ybf5ufhagycj5g7hq5vi&dl=1",
        "GSE127064_AdBrain_gex2atac":
        "https://www.dropbox.com/scl/fi/mktue5y4bsf9w17t7jyq3/GSE127064_AdBrain_gex2atac.zip?rlkey=3qtazuova6v1rin630keryman&dl=1",
        "GSE127064_p0Brain_gex2atac":
        "https://www.dropbox.com/scl/fi/anlukciivt5ah4i9v5q8s/GSE127064_p0Brain_gex2atac.zip?rlkey=9q12rwqgbz2z45dkwz372grgk&dl=1",
        "GSE140203_3T3_HG19_atac2gex":
        "https://www.dropbox.com/scl/fi/840hsqkcbis0t35i04kdi/GSE140203_3T3_HG19_atac2gex.zip?rlkey=gurncv741zi4q6dqb9q293zsl&dl=1",
        "GSE140203_3T3_MM10_atac2gex":
        "https://www.dropbox.com/scl/fi/chtl13dchlteilm2hky7r/GSE140203_3T3_MM10_atac2gex.zip?rlkey=su1itxejsyzkqcxjngb1xbunj&dl=1",
        "GSE140203_12878.rep2_atac2gex":
        "https://www.dropbox.com/scl/fi/9axnm23b554tn7uenf98q/GSE140203_12878.rep2_atac2gex.zip?rlkey=dplthpb82qhvnh9fann5o1gvb&dl=1",
        "GSE140203_12878.rep3_atac2gex":
        "https://www.dropbox.com/scl/fi/1zgc35dbl1pyrwrqfmtj8/GSE140203_12878.rep3_atac2gex.zip?rlkey=lwkx6iv2z584m1315gqpcomw9&dl=1",
        "GSE140203_K562_HG19_atac2gex":
        "https://www.dropbox.com/scl/fi/kro3384oium84fdr46l77/GSE140203_K562_HG19_atac2gex.zip?rlkey=f9kyx8rz4o7tgf8vts64d5rpi&dl=1",
        "GSE140203_K562_MM10_atac2gex":
        "https://www.dropbox.com/scl/fi/2dwn8zzhaq86ojkfgh29q/GSE140203_K562_MM10_atac2gex.zip?rlkey=ek94g5d9w0xrafp72z9jx5wty&dl=1",
        "GSE140203_LUNG_atac2gex":
        "https://www.dropbox.com/scl/fi/zb7igtgg835pg73ec7f28/GSE140203_LUNG_atac2gex.zip?rlkey=19ohnkxpj1temqnfxje6jae8w&dl=1",
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
        if is_numeric(train_mod2.obs_names[0]):
            train_mod2.obs_names = train_mod1.obs_names
        if is_numeric(test_mod2.obs_names[0]):
            test_mod2.obs_names = test_mod1.obs_names

        # TODO: support other two subtasks
        # assert self.subtask in ("openproblems_bmmc_cite_phase2_rna", "openproblems_bmmc_cite_phase2_rna_subset",
        #                         "openproblems_bmmc_multiome_phase2_rna","pbmc_cite","openproblems_2022_multi_atac2gex","openproblems_2022_cite_gex2adt"), "Currently not available."
        changed_count = 0  # keep track to modified entries due to ensuring count data type
        for i in range(4):
            m_data = modalities[i].X
            int_data = m_data.astype(int)
            changed_count += np.sum(int_data != m_data)
            modalities[i].X = int_data
            modalities[i].layers["counts"] = modalities[i].X
        if changed_count > 0:
            logger.warning("Implicit modification: to ensure count (integer type) data, "
                           f"a total number of {changed_count} entries were modified.")

        if self.preprocess == "pca":
            if self.pkl_path and osp.exists(self.pkl_path):
                with open(self.pkl_path, "rb") as f:
                    preprocessed_features = pickle.load(f)

            else:
                for i in range(2):
                    sc.pp.filter_genes(modalities[i], min_cells=1, inplace=True)
                    sc.pp.filter_genes(modalities[i + 2], min_cells=1, inplace=True)
                    common_genes = list(set(modalities[i].var.index) & set(modalities[i + 2].var.index))
                    modalities[i] = modalities[i][:, common_genes]
                    modalities[i + 2] = modalities[i + 2][:, common_genes]
                if self.subtask in ("openproblems_2022_cite_gex2adt_subset", "pbmc_cite",
                                    "openproblems_bmmc_cite_phase2_rna", "openproblems_bmmc_cite_phase2_rna_subset",
                                    "5k_pbmc_subset"):
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(modalities[2]).values
                    m2_train = modalities[1].X.toarray()
                    m2_test = modalities[3].X.toarray()
                elif self.subtask in ("GSE117089_mouse_gex2atac", "GSE117089_sciCAR_gex2atac",
                                      "GSE127064_AdBrain_gex2atac", "GSE127064_p0Brain_gex2atac",
                                      "openproblems_bmmc_multiome_phase2_rna", "10k_pbmc"):
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(modalities[2]).values
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    m2_train = lsi_transformer_atac.fit_transform(modalities[1]).values
                    m2_test = lsi_transformer_atac.transform(modalities[3]).values
                elif self.subtask in ("openproblems_2022_multi_atac2gex_subset", "GSE140203_3T3_HG19_atac2gex",
                                      "GSE140203_3T3_MM10_atac2gex", "GSE140203_12878.rep2_atac2gex",
                                      "GSE140203_12878.rep3_atac2gex", "GSE140203_K562_HG19_atac2gex",
                                      "GSE140203_K562_MM10_atac2gex", "GSE140203_LUNG_atac2gex"):
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    m1_train = lsi_transformer_atac.fit_transform(modalities[0]).values
                    m1_test = lsi_transformer_atac.transform(modalities[2]).values
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m2_train = lsi_transformer_gex.fit_transform(modalities[1]).values
                    m2_test = lsi_transformer_gex.transform(modalities[3]).values
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
            for i in [0, 2]:
                sc.pp.filter_cells(modalities[i], min_counts=1, inplace=True)
                sc.pp.filter_cells(modalities[i + 1], min_counts=1, inplace=True)
                common_cells = list(set(modalities[i].obs.index) & set(modalities[i + 1].obs.index))
                modalities[i] = modalities[i][common_cells, :]
                modalities[i + 1] = modalities[i + 1][common_cells, :]
                if i == 0:
                    train_label = train_label[common_cells, :]
                    train_label = ad.AnnData(obs=train_label.obs, X=sp.csr_matrix(np.eye(len(train_label.obs))))
                else:
                    test_label = test_label[common_cells, :]
                    test_label = ad.AnnData(obs=test_label.obs, X=sp.csr_matrix(np.eye(len(test_label.obs))))
            for i in range(2):
                if modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(modalities[i], layer="counts", flavor="seurat_v3",
                                                n_top_genes=selection_threshold, span=self.span)
                    modalities[i + 2].var["highly_variable"] = modalities[i].var["highly_variable"]
                    modalities[i] = modalities[i][:, modalities[i].var["highly_variable"]]
                    modalities[i + 2] = modalities[i + 2][:, modalities[i + 2].var["highly_variable"]]
            for i in [0, 2]:
                sc.pp.filter_cells(modalities[i], min_counts=1, inplace=True)
                sc.pp.filter_cells(modalities[i + 1], min_counts=1, inplace=True)
                common_cells = list(set(modalities[i].obs.index) & set(modalities[i + 1].obs.index))
                modalities[i] = modalities[i][common_cells, :]
                modalities[i + 1] = modalities[i + 1][common_cells, :]
                if i == 0:
                    train_label = train_label[common_cells, :]
                    train_label = ad.AnnData(obs=train_label.obs, X=sp.csr_matrix(np.eye(len(train_label.obs))))
                else:
                    test_label = test_label[common_cells, :]
                    test_label = ad.AnnData(obs=test_label.obs, X=sp.csr_matrix(np.eye(len(test_label.obs))))

        else:
            logger.info("Preprocessing method not supported.")

        logger.info("Preprocessing done.")

        train_mod1, train_mod2, test_mod1, test_mod2 = modalities
        return train_mod1, train_mod2, train_label, test_mod1, test_mod2, test_label


@register_dataset("multimodality")
class JointEmbeddingNIPSDataset(MultiModalityDataset):

    TASK = "joint_embedding"
    URL_DICT = {
        "openproblems_bmmc_cite_phase2":
        "https://www.dropbox.com/s/hjr4dxuw55vin5z/openproblems_bmmc_cite_phase2.zip?dl=1",
        "openproblems_bmmc_multiome_phase2":
        "https://www.dropbox.com/s/40kjslupxhkg92s/openproblems_bmmc_multiome_phase2.zip?dl=1",
        "GSE140203_BRAIN_atac2gex":
        "https://www.dropbox.com/scl/fi/pa4zpj1fp00cqiavjadtx/GSE140203_BRAIN_atac2gex.zip?rlkey=oy73h59w4p9jsyhoxtaerxfw5&dl=1",
        "GSE140203_SKIN_atac2gex":
        "https://www.dropbox.com/scl/fi/2yuatq0icu6g5pc37jxq7/GSE140203_SKIN_atac2gex.zip?rlkey=o9fzlogrk3thcycv6u20jbyc6&dl=1",
        "openproblems_2022_cite_gex2adt":
        "https://www.dropbox.com/scl/fi/j3att18aems8ve8qhykeu/openproblems_2022_cite_gex2adt.zip?rlkey=i85wjp8iqkpxhbknywmwz8mz6&dl=1",
        "openproblems_2022_multi_atac2gex":
        "https://www.dropbox.com/scl/fi/fcw493eef1kmegwh9dpq9/openproblems_2022_multi_atac2gex.zip?rlkey=sd0dxbb9iadj84f84ai5cm5q5&dl=1"
    }
    SUBTASK_NAME_MAP = {
        "adt": "openproblems_bmmc_cite_phase2",
        "atac": "openproblems_bmmc_multiome_phase2",
    }
    AVAILABLE_DATA = sorted(list(URL_DICT) + list(SUBTASK_NAME_MAP))

    def __init__(self, subtask, root="./data", preprocess=None, normalize=False, pretrained_folder=".",
                 selection_threshold=10000, span=0.3):
        # TODO: factor our preprocess
        self.preprocess = preprocess
        self.normalize = normalize
        self.pretrained_folder = pretrained_folder
        super().__init__(subtask, root)
        self.selection_threshold = selection_threshold
        self.span = span

    def _raw_to_dance(self, raw_data):
        mod1, mod2, meta1, meta2, test_sol = self._maybe_preprocess(raw_data)

        assert all(mod2.obs_names == mod1.obs_names), "Modalities not aligned"
        mdata = md.MuData({"mod1": mod1, "mod2": mod2, "meta1": meta1, "meta2": meta2, "test_sol": test_sol})

        train_size = meta1.shape[0]
        data = Data(mdata, train_size=train_size)

        return data

    def _maybe_preprocess(self, raw_data):
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
        elif self.preprocess == "pca":
            sc.pp.filter_genes(mod1, min_counts=3)
            sc.pp.filter_genes(mod2, min_counts=3)
            meta1 = meta1[:, mod1.var.index]
            meta2 = meta2[:, mod2.var.index]
            test_sol = test_sol[:, mod1.var.index]
            lsi_transformer_gex = lsiTransformer(n_components=64, drop_first=True)
            mod1.obsm['X_pca'] = lsi_transformer_gex.fit_transform(mod1).values
            mod2.obsm['X_pca'] = lsi_transformer_gex.fit_transform(mod2).values

        elif self.preprocess == "feature_selection":

            sc.pp.filter_genes(mod1, min_counts=3)
            sc.pp.filter_genes(mod2, min_counts=3)
            meta1 = meta1[:, mod1.var.index]
            meta2 = meta2[:, mod2.var.index]
            test_sol = test_sol[:, mod1.var.index]

            if mod1.shape[1] > self.selection_threshold:
                sc.pp.highly_variable_genes(mod1, layer="counts", flavor="seurat_v3",
                                            n_top_genes=self.selection_threshold, span=self.span)
                mod1 = mod1[:, mod1.var["highly_variable"]]
            if mod2.shape[1] > self.selection_threshold:
                sc.pp.highly_variable_genes(mod2, layer="counts", flavor="seurat_v3",
                                            n_top_genes=self.selection_threshold, span=self.span)
                mod2 = mod2[:, mod2.var["highly_variable"]]
            sc.pp.filter_cells(mod1, min_genes=1, inplace=True)
            sc.pp.filter_cells(mod2, min_genes=1, inplace=True)
            common_cells = list(set(mod1.obs.index) & set(mod2.obs.index))
            mod1 = mod1[common_cells, :]
            mod2 = mod2[common_cells, :]
            test_sol = test_sol[common_cells, :]

            sc.pp.filter_cells(meta1, min_genes=1, inplace=True)
            sc.pp.filter_cells(meta2, min_genes=1, inplace=True)
            meta_common_cells = list(set(meta1.obs.index) & set(meta2.obs.index))
            meta1 = meta1[meta_common_cells, :]
            meta2 = meta2[meta_common_cells, :]
        else:
            logger.info(f"Preprocessing method {self.preprocess!r} not supported.")

        # Normalization
        if self.normalize:
            sc.pp.scale(mod1)
            sc.pp.scale(mod2)

        logger.info("Preprocessing done.")

        return mod1, mod2, meta1, meta2, test_sol
