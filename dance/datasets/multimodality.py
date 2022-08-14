import os
import pickle

import anndata as ad
import numpy as np
import scanpy as sc
import torch

from dance.data import *
from dance.transforms.preprocess import lsiTransformer


class MultiModalityDataset():

    def __init__(self, task, data_url, subtask, data_dir="./data"):

        assert (subtask in [
            'openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
            'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2', 'openproblems_bmmc_cite_phase2',
            'openproblems_bmmc_multiome_phase2', 'adt2gex', 'gex2adt', 'atac2gex', 'gex2atac'
        ]), 'Undefined subtask.'

        assert (task in ['predict_modality', 'match_modality', 'joint_embedding']), 'Undefined task.'

        # regularize subtask name
        if task == 'joint_embedding':
            if subtask == 'adt':
                subtask = 'openproblems_bmmc_cite_phase2'
            elif subtask == 'atac':
                subtask = 'openproblems_bmmc_multiome_phase2'
        else:
            if subtask == 'adt2gex':
                subtask = 'openproblems_bmmc_cite_phase2_mod2'
            elif subtask == 'gex2adt':
                subtask = 'openproblems_bmmc_cite_phase2_rna'
            elif subtask == 'atac2gex':
                subtask = 'openproblems_bmmc_multiome_phase2_mod2'
            elif subtask == 'gex2atac':
                subtask = 'openproblems_bmmc_multiome_phase2_rna'

        self.task = task
        self.subtask = subtask
        self.data_dir = data_dir
        self.loaded = False
        self.data_url = data_url

    def download_data(self):
        # download data
        download_file(self.data_url, self.data_dir + "/{}.zip".format(self.subtask))
        unzip_file(self.data_dir + "/{}.zip".format(self.subtask), self.data_dir)
        return self

    def download_pathway(self):
        download_file('https://www.dropbox.com/s/uqoakpalr3albiq/h.all.v7.4.entrez.gmt?dl=1',
                      self.data_dir + "/h.all.v7.4.entrez.gmt")
        download_file('https://www.dropbox.com/s/yjrcsd2rpmahmfo/h.all.v7.4.symbols.gmt?dl=1',
                      self.data_dir + "/h.all.v7.4.symbols.gmt")
        return self

    def is_complete(self):
        # judge data is complete or not
        if self.task == 'joint_embedding':
            return os.path.exists(
                os.path.join(
                    self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod1.h5ad')) and os.path.exists(
                        os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod2.h5ad'))
        else:
            return os.path.exists(
                os.path.join(
                    self.data_dir, self.subtask,
                    f'{self.subtask}.censor_dataset.output_train_mod1.h5ad')) and os.path.exists(
                        os.path.join(
                            self.data_dir, self.subtask,
                            f'{self.subtask}.censor_dataset.output_train_mod2.h5ad')) and os.path.exists(
                                os.path.join(
                                    self.data_dir, self.subtask,
                                    f'{self.subtask}.censor_dataset.output_test_mod1.h5ad')) and os.path.exists(
                                        os.path.join(self.data_dir, self.subtask,
                                                     f'{self.subtask}.censor_dataset.output_test_mod2.h5ad'))

    def load_data(self):
        # Load data from existing h5ad files, or download files and load data.
        if self.is_complete():
            pass
        else:
            self.download_data()
            assert self.is_complete()

        if self.task == 'joint_embedding':
            mod_path_list = [
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_mod2.h5ad')
            ]
        else:
            mod_path_list = [
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_mod2.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod1.h5ad'),
                os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_mod2.h5ad')
            ]

        self.modalities = []
        for mod_path in mod_path_list:
            self.modalities.append(ad.read_h5ad(mod_path))
        self.loaded = True
        return self

    def sparse_features(self, index=None, count=False):
        assert self.loaded, 'Data have not been loaded.'
        if not count:
            if index is None:
                return [mod.X for mod in self.modalities]
            else:
                return self.modalities[index].X
        else:
            if index is None:
                return [mod.layers['counts'] for mod in self.modalities]
            else:
                return self.modalities[index].layers['counts']

    def numpy_features(self, index=None, count=False):
        assert self.loaded, 'Data have not been loaded.'
        if not count:
            if index is None:
                return [mod.X.toarray() for mod in self.modalities]
            else:
                return self.modalities[index].X.toarray()
        else:
            if index is None:
                return [mod.layers['counts'].toarray() for mod in self.modalities]
            else:
                return self.modalities[index].layers['counts'].toarray()

    def tensor_features(self, index=None, count=False, device='cpu'):
        assert self.loaded, 'Data have not been loaded.'
        if not count:
            if index is None:
                return [torch.from_numpy(mod.X.toarray()).to(device) for mod in self.modalities]
            else:
                return torch.from_numpy(self.modalities[index].X.toarray()).to(device)
        else:
            if index is None:
                return [torch.from_numpy(mod.layers['counts'].toarray()).to(device) for mod in self.modalities]
            else:
                return torch.from_numpy(self.modalities[index].layers['counts'].toarray()).to(device)

    def get_modalities(self):
        assert self.loaded, 'Data have not been loaded.'
        return self.modalities


class ModalityPredictionDataset(MultiModalityDataset):

    def __init__(self, subtask, data_dir="./data"):
        assert (subtask in [
            'openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
            'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2', 'adt2gex', 'gex2adt', 'atac2gex',
            'gex2atac'
        ]), 'Undefined subtask.'

        if subtask == 'adt2gex':
            subtask = 'openproblems_bmmc_cite_phase2_mod2'
        elif subtask == 'gex2adt':
            subtask = 'openproblems_bmmc_cite_phase2_rna'
        elif subtask == 'atac2gex':
            subtask = 'openproblems_bmmc_multiome_phase2_mod2'
        elif subtask == 'gex2atac':
            subtask = 'openproblems_bmmc_multiome_phase2_rna'

        data_url = {
            'openproblems_bmmc_cite_phase2_mod2':
            'https://www.dropbox.com/s/snh8knscnlcq4um/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
            'openproblems_bmmc_cite_phase2_rna':
            'https://www.dropbox.com/s/xbfyhv830u9pupv/openproblems_bmmc_cite_phase2_rna.zip?dl=1',
            'openproblems_bmmc_multiome_phase2_mod2':
            'https://www.dropbox.com/s/p9ve2ljyy4yqna4/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1',
            'openproblems_bmmc_multiome_phase2_rna':
            'https://www.dropbox.com/s/cz60vp7bwapz0kw/openproblems_bmmc_multiome_phase2_rna.zip?dl=1'
        }

        super().__init__('predict_modality', data_url.get(subtask), subtask, data_dir)

    def preprocess(self, kind='feature_selection', selection_threshold=10000):
        if kind == 'pca':
            print('Preprocessing method not supported.')
            return self
        elif kind == 'feature_selection':
            if self.modalities[0].shape[1] > selection_threshold:
                sc.pp.highly_variable_genes(self.modalities[0], layer='counts', flavor='seurat_v3',
                                            n_top_genes=selection_threshold)
                self.modalities[2].var['highly_variable'] = self.modalities[0].var['highly_variable']
                for i in [0, 2]:
                    self.modalities[i] = self.modalities[i][:, self.modalities[i].var['highly_variable']]
        else:
            print('Preprocessing method not supported.')
            return self
        print('Preprocessing done.')
        return self


class ModalityMatchingDataset(MultiModalityDataset):

    def __init__(self, subtask, data_dir="./data"):
        assert (subtask in [
            'openproblems_bmmc_multiome_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna',
            'openproblems_bmmc_cite_phase2_rna', 'openproblems_bmmc_cite_phase2_mod2', 'adt2gex', 'gex2adt', 'atac2gex',
            'gex2atac'
        ]), 'Undefined subtask.'

        if subtask == 'adt2gex':
            subtask = 'openproblems_bmmc_cite_phase2_mod2'
        elif subtask == 'gex2adt':
            subtask = 'openproblems_bmmc_cite_phase2_rna'
        elif subtask == 'atac2gex':
            subtask = 'openproblems_bmmc_multiome_phase2_mod2'
        elif subtask == 'gex2atac':
            subtask = 'openproblems_bmmc_multiome_phase2_rna'

        data_url = {
            'openproblems_bmmc_cite_phase2_mod2':
            'https://www.dropbox.com/s/fa6zut89xx73itz/openproblems_bmmc_cite_phase2_mod2.zip?dl=1',
            'openproblems_bmmc_cite_phase2_rna':
            'https://www.dropbox.com/s/ep00mqcjmdu0b7v/openproblems_bmmc_cite_phase2_rna.zip?dl=1',
            'openproblems_bmmc_multiome_phase2_mod2':
            'https://www.dropbox.com/s/31qi5sckx768acw/openproblems_bmmc_multiome_phase2_mod2.zip?dl=1',
            'openproblems_bmmc_multiome_phase2_rna':
            'https://www.dropbox.com/s/h1s067wkefs1jh2/openproblems_bmmc_multiome_phase2_rna.zip?dl=1'
        }

        super().__init__('match_modality', data_url.get(subtask), subtask, data_dir)
        self.preprocessed = False

    def load_sol(self):
        assert (self.loaded)
        self.train_sol = ad.read_h5ad(
            os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_train_sol.h5ad'))
        self.test_sol = ad.read_h5ad(
            os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_test_sol.h5ad'))
        self.modalities[1] = self.modalities[1][self.train_sol.to_df().values.argmax(1)]
        return self

    def preprocess(self, kind='pca', pkl_path=None, selection_threshold=10000):

        # TODO: support other two subtasks
        assert self.subtask in ('openproblems_bmmc_cite_phase2_rna',
                                'openproblems_bmmc_multiome_phase2_rna'), 'Currently not available.'

        if kind == 'pca':
            if pkl_path and (not os.path.exists(pkl_path)):

                if self.subtask == 'openproblems_bmmc_cite_phase2_rna':
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                    m2_train = self.modalities[1].X.toarray()
                    m2_test = self.modalities[3].X.toarray()

                if self.subtask == 'openproblems_bmmc_multiome_phase2_rna':
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    m1_train = lsi_transformer_gex.fit_transform(self.modalities[0]).values
                    m1_test = lsi_transformer_gex.transform(self.modalities[2]).values
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    m2_train = lsi_transformer_atac.fit_transform(self.modalities[1]).values
                    m2_test = lsi_transformer_atac.transform(self.modalities[3]).values

                self.preprocessed_features = {
                    'mod1_train': m1_train,
                    'mod2_train': m2_train,
                    'mod1_test': m1_test,
                    'mod2_test': m2_test
                }
                pickle.dump(self.preprocessed_features, open(pkl_path, 'wb'))

            else:
                self.preprocessed_features = pickle.load(open(pkl_path, 'rb'))
        elif kind == 'feature_selection':
            for i in range(2):
                if self.modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(self.modalities[i], layer='counts', flavor='seurat_v3',
                                                n_top_genes=selection_threshold)
                    self.modalities[i + 2].var['highly_variable'] = self.modalities[i].var['highly_variable']
                    self.modalities[i] = self.modalities[i][:, self.modalities[i].var['highly_variable']]
                    self.modalities[i + 2] = self.modalities[i + 2][:, self.modalities[i + 2].var['highly_variable']]
        else:
            print('Preprocessing method not supported.')
            return self
        print('Preprocessing done.')
        self.preprocessed = True
        return self

    def get_preprocessed_features(self):
        assert self.preprocessed, 'Transformed features do not exist.'
        return self.preprocessed_features


class JointEmbeddingNIPSDataset(MultiModalityDataset):

    def __init__(self, subtask, data_dir="./data"):
        assert (subtask in ['openproblems_bmmc_multiome_phase2', 'openproblems_bmmc_cite_phase2', 'adt',
                            'atac']), 'Undefined subtask.'

        if subtask == 'adt':
            subtask = 'openproblems_bmmc_cite_phase2'
        elif subtask == 'atac':
            subtask = 'openproblems_bmmc_multiome_phase2'

        data_url = {
            'openproblems_bmmc_cite_phase2':
            'https://www.dropbox.com/s/hjr4dxuw55vin5z/openproblems_bmmc_cite_phase2.zip?dl=1',
            'openproblems_bmmc_multiome_phase2':
            'https://www.dropbox.com/s/40kjslupxhkg92s/openproblems_bmmc_multiome_phase2.zip?dl=1'
        }
        super().__init__('joint_embedding', data_url.get(subtask), subtask, data_dir)
        self.preprocessed = False

    def load_metadata(self):
        assert (self.loaded)

        if self.subtask.find('cite') != -1:
            mod = 'adt'
            meta = 'cite'
        else:
            mod = 'atac'
            meta = 'multiome'
        self.exploration = [
            ad.read_h5ad(os.path.join(self.data_dir, self.subtask, f'{meta}_gex_processed_training.h5ad')),
            ad.read_h5ad(os.path.join(self.data_dir, self.subtask, f'{meta}_{mod}_processed_training.h5ad')),
        ]
        return self

    def load_sol(self):
        assert (self.loaded)
        self.test_sol = ad.read_h5ad(
            os.path.join(self.data_dir, self.subtask, f'{self.subtask}.censor_dataset.output_solution.h5ad'))
        return self

    def preprocess(self, kind='aux', pretrained_folder='.', selection_threshold=10000):
        if kind == 'aux':
            os.makedirs(pretrained_folder, exist_ok=True)

            if os.path.exists(os.path.join(pretrained_folder, f'preprocessed_data_{self.subtask}.pkl')):

                with open(os.path.join(pretrained_folder, f'preprocessed_data_{self.subtask}.pkl'), 'rb') as f:
                    self.preprocessed_data = pickle.load(f)
                with open(os.path.join(pretrained_folder, f'{self.subtask}_config.pk'), 'rb') as f:
                    self.nb_cell_types, self.nb_batches, self.nb_phases = pickle.load(f)
                self.preprocessed = True
                print('Preprocessing done.')
                return self

            ##########################################
            ##             PCA PRETRAIN             ##
            ##########################################

            # scale and log transform
            scale = 1e4
            n_components_mod1, n_components_mod2 = 256, 100

            mod1 = self.modalities[0].var["feature_types"][0]
            mod2 = self.modalities[1].var["feature_types"][0]

            if mod2 == "ADT":
                if os.path.exists(os.path.join(pretrained_folder, f"lsi_cite_{mod1}.pkl")):
                    lsi_transformer_gex = pickle.load(
                        open(os.path.join(pretrained_folder, f"lsi_cite_{mod1}.pkl"), "rb"))
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=256, drop_first=True)
                    lsi_transformer_gex.fit(self.modalities[0])
                    pickle.dump(lsi_transformer_gex, open(os.path.join(pretrained_folder, f"lsi_cite_{mod1}.pkl"),
                                                          "wb"))

            if mod2 == "ATAC":

                if os.path.exists(os.path.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl")):
                    with open(os.path.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl"), "rb") as f:
                        lsi_transformer_gex = pickle.load(f)
                else:
                    lsi_transformer_gex = lsiTransformer(n_components=64, drop_first=True)
                    lsi_transformer_gex.fit(self.modalities[0])
                    with open(os.path.join(pretrained_folder, f"lsi_multiome_{mod1}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_gex, f)

                if os.path.exists(os.path.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl")):
                    with open(os.path.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl"), "rb") as f:
                        lsi_transformer_atac = pickle.load(f)
                else:
                    #         lsi_transformer_atac = TruncatedSVD(n_components=100, random_state=random_seed)
                    lsi_transformer_atac = lsiTransformer(n_components=512, drop_first=True)
                    lsi_transformer_atac.fit(self.modalities[1])
                    with open(os.path.join(pretrained_folder, f"lsi_multiome_{mod2}.pkl"), "wb") as f:
                        pickle.dump(lsi_transformer_atac, f)

            ##########################################
            ##           DATA PREPROCESSING         ##
            ##########################################

            ad_mod1 = self.exploration[0]
            ad_mod2 = self.exploration[1]
            mod1_obs = ad_mod1.obs

            # Make sure exploration data match the full data
            assert ((self.modalities[0].obs['batch'].index[:mod1_obs.shape[0]] == mod1_obs['batch'].index).mean() == 1)

            if mod2 == "ADT":
                mod1_pca = lsi_transformer_gex.transform(ad_mod1).values
                mod1_pca_test = lsi_transformer_gex.transform(self.modalities[0][mod1_obs.shape[0]:]).values
                mod2_pca = ad_mod2.X.toarray()
                mod2_pca_test = self.numpy_features(1)[mod1_obs.shape[0]:]

            if mod2 == "ATAC":
                mod1_pca = lsi_transformer_gex.transform(ad_mod1).values
                mod1_pca_test = lsi_transformer_gex.transform(self.modalities[0][mod1_obs.shape[0]:]).values
                mod2_pca = lsi_transformer_atac.transform(ad_mod2).values
                mod2_pca_test = lsi_transformer_atac.transform(self.modalities[1][mod1_obs.shape[0]:]).values

            cell_cycle_genes = [
                'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', \
                'MCM4', 'RRM1', 'UNG', 'GINS2', 'MCM6', \
                'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', \
                'HELLS', 'RFC2', 'RPA2', 'NASP', 'RAD51AP1', \
                'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', \
                'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', \
                'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', \
                'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8', \
                'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', \
                'TOP2A', 'NDC80', 'CKS2', 'NUF2', 'CKS1B', 'MKI67', \
                'TMPO', 'CENPF', 'TACC3', 'FAM64A', 'SMC4', 'CCNB2', \
                'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', \
                'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', \
                'CDC20', 'TTK', 'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', \
                'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2', 'KIF23', 'HMMR', \
                'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 'CTCF', \
                'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA']

            print('Data loading and pca done', mod1_pca.shape, mod2_pca.shape)
            print('Start to calculate cell_cycle score. It may roughly take an hour.')

            pca_combined = np.concatenate([mod1_pca, mod2_pca], axis=1)

            del mod1_pca, mod2_pca

            cell_type_labels = mod1_obs['cell_type']
            batch_ids = mod1_obs['batch']
            phase_labels = mod1_obs['phase']
            nb_cell_types = len(np.unique(cell_type_labels))
            nb_batches = len(np.unique(batch_ids))
            nb_phases = len(np.unique(phase_labels)) - 1  # 2
            c_labels = np.array([list(np.unique(cell_type_labels)).index(item) for item in cell_type_labels])
            b_labels = np.array([list(np.unique(batch_ids)).index(item) for item in batch_ids])
            p_labels = np.array([list(np.unique(phase_labels)).index(item) for item in phase_labels])
            # 0:G1, 1:G2M, 2: S, only consider the last two
            s_genes = cell_cycle_genes[:43]
            g2m_genes = cell_cycle_genes[43:]
            sc.pp.log1p(ad_mod1)
            sc.pp.scale(ad_mod1)
            sc.tl.score_genes_cell_cycle(ad_mod1, s_genes=s_genes, g2m_genes=g2m_genes)
            S_scores = ad_mod1.obs['S_score'].values
            G2M_scores = ad_mod1.obs['G2M_score'].values
            phase_scores = np.stack([S_scores, G2M_scores]).T  # (nb_cells, 2)

            X_train = pca_combined
            # c_labels: cell type; b_labels: batch ids; p_labels: phase_labels; phase_scores
            Y_train = [c_labels, b_labels, p_labels, phase_scores]
            X_test = np.concatenate([mod1_pca_test, mod2_pca_test], axis=1)

            self.preprocessed_data = {'X_train': X_train, 'Y_train': Y_train, 'X_test': X_test}
            pickle.dump(self.preprocessed_data,
                        open(os.path.join(pretrained_folder, f'preprocessed_data_{self.subtask}.pkl'), 'wb'))
            pickle.dump([nb_cell_types, nb_batches, nb_phases],
                        open(os.path.join(pretrained_folder, f'{self.subtask}_config.pk'), 'wb'))

            self.nb_cell_types, self.nb_batches, self.nb_phases = nb_cell_types, nb_batches, nb_phases
        elif kind == 'feature_selection':
            for i in range(2):
                if self.modalities[i].shape[1] > selection_threshold:
                    sc.pp.highly_variable_genes(self.modalities[i], layer='counts', flavor='seurat_v3',
                                                n_top_genes=selection_threshold)
                    self.modalities[i] = self.modalities[i][:, self.modalities[i].var['highly_variable']]
        else:
            print('Preprocessing method not supported.')
            return self
        self.preprocessed = True
        print('Preprocessing done.')
        return self

    def get_preprocessed_data(self):
        return self.preprocessed_data

    def normalize(self):
        assert self.preprocessed, 'Normalization must be conducted after preprocessing.'

        self.mean = np.concatenate([self.preprocessed_data['X_train'], self.preprocessed_data['X_test']], 0).mean()
        self.std = np.concatenate([self.preprocessed_data['X_train'], self.preprocessed_data['X_test']], 0).std()
        self.preprocessed_data['X_train'] = (self.preprocessed_data['X_train'] - self.mean) / self.std
        self.preprocessed_data['X_test'] = (self.preprocessed_data['X_test'] - self.mean) / self.std

        return self
