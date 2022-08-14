import argparse
import os
import random
from argparse import Namespace

import dance.transforms.preprocess
from dance.datasets.multimodality import ModalityPredictionDataset
from dance.modules.multi_modality.predict_modality.scmogcn import *
from dance.transforms.graph_construct import construct_modality_prediction_graph, gen_batch_features
from dance.utils import set_seed


def pipeline(transductive=True, verbose=2, logger=None, **kwargs):
    PREFIX = kwargs['prefix']
    os.makedirs(kwargs["log_folder"], exist_ok=True)
    os.makedirs(kwargs["model_folder"], exist_ok=True)
    os.makedirs(kwargs["result_folder"], exist_ok=True)
    if verbose > 1:
        logger = open(f'{kwargs["log_folder"]}/{PREFIX}.log', 'w')
        logger.write(str(kwargs) + '\n')

    subtask = kwargs['subtask']

    dataset = ModalityPredictionDataset(subtask).load_data()
    dataset.download_pathway()
    if kwargs['preprocessing'] != 'none':
        dataset = dataset.preprocess(kwargs['preprocessing'])

    idx = np.random.permutation(dataset.modalities[0].shape[0])
    split = {'train': idx[:-int(len(idx) * 0.15)], 'valid': idx[-int(len(idx) * 0.15):]}

    input_train_mod1 = dataset.sparse_features()[0]
    input_train_mod2 = dataset.sparse_features()[1]
    if transductive:
        input_test_mod1 = dataset.sparse_features()[2]
        true_test_mod2 = dataset.sparse_features()[3]

    FEATURE_SIZE = input_train_mod1.shape[1]
    CELL_SIZE = input_train_mod1.shape[0] + input_test_mod1.shape[0] if transductive else input_train_mod1.shape[0]
    OUTPUT_SIZE = input_train_mod2.shape[1]
    TRAIN_SIZE = input_train_mod1.shape[0]

    kwargs['FEATURE_SIZE'] = FEATURE_SIZE
    kwargs['CELL_SIZE'] = CELL_SIZE
    kwargs['OUTPUT_SIZE'] = OUTPUT_SIZE
    kwargs['TRAIN_SIZE'] = TRAIN_SIZE

    if kwargs['inductive'] != 'trans':
        g, gtest = construct_modality_prediction_graph(dataset, **kwargs)
    else:
        gtest = g = construct_modality_prediction_graph(dataset, **kwargs)

    if not kwargs['no_batch_features']:
        if transductive:
            batch_features = gen_batch_features([dataset.modalities[0], dataset.modalities[2]])
        else:
            batch_features = gen_batch_features(dataset.modalities[:1])
        kwargs['BATCH_NUM'] = batch_features.shape[1]
        if kwargs['inductive'] != 'trans':
            g.nodes['cell'].data['bf'] = batch_features[:TRAIN_SIZE]
            gtest.nodes['cell'].data['bf'] = batch_features
        else:
            g.nodes['cell'].data['bf'] = batch_features

    device = kwargs['device']
    g = g.to(device)
    if kwargs['inductive'] != 'trans':
        gtest = gtest.to(device)

    # data loader
    y = torch.from_numpy(input_train_mod2.toarray()).to(device)
    if transductive:
        y_test = torch.from_numpy(true_test_mod2.toarray()).to(device)

    model = ScMoGCNWrapper(Namespace(**kwargs))

    if kwargs['sampling']:
        model.fit_with_sampling(g, y, split, transductive, verbose, y_test, logger)
    else:
        model.fit(g, y, split, transductive, verbose, y_test, logger)

    print(model.predict(g, np.arange(TRAIN_SIZE, CELL_SIZE)))
    print(model.score(g, np.arange(TRAIN_SIZE, CELL_SIZE), y_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-prefix', '--prefix', default='dance_openproblems_bmmc_atac2rna_test')
    parser.add_argument('-t', '--subtask', default='openproblems_bmmc_cite_phase2_rna')
    parser.add_argument('-pww', '--pathway_weight', default='pearson', choices=['cos', 'one', 'pearson'])
    parser.add_argument('-pwth', '--pathway_threshold', type=float, default=-1.0)
    parser.add_argument('-l', '--log_folder', default='./logs')
    parser.add_argument('-m', '--model_folder', default='./models')
    parser.add_argument('-r', '--result_folder', default='./results')
    parser.add_argument('-e', '--epoch', type=int, default=10000)
    parser.add_argument('-nbf', '--no_batch_features', action='store_true')
    parser.add_argument('-npw', '--no_pathway', action='store_true')
    parser.add_argument('-opw', '--only_pathway', action='store_true')
    parser.add_argument('-res', '--residual', default='res_cat', choices=['none', 'res_add', 'res_cat'])
    parser.add_argument('-inres', '--initial_residual', action='store_true')
    parser.add_argument('-pwagg', '--pathway_aggregation', default='alpha',
                        choices=['sum', 'attention', 'two_gate', 'one_gate', 'alpha', 'cat'])
    parser.add_argument('-pwalpha', '--pathway_alpha', type=float, default=0.5)
    parser.add_argument('-nrc', '--no_readout_concatenate', action='store_true')
    parser.add_argument('-bs', '--batch_size', default=1000, type=int)
    parser.add_argument('-nm', '--normalization', default='group', choices=['batch', 'layer', 'group', 'none'])
    parser.add_argument('-ac', '--activation', default='gelu', choices=['leaky_relu', 'relu', 'prelu', 'gelu'])
    parser.add_argument('-em', '--embedding_layers', default=1, type=int, choices=[1, 2, 3])
    parser.add_argument('-ro', '--readout_layers', default=1, type=int, choices=[1, 2])
    parser.add_argument('-conv', '--conv_layers', default=4, type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('-agg', '--agg_function', default='mean', choices=['gcn', 'mean'])
    parser.add_argument('-device', '--device', default='cuda')
    parser.add_argument('-sb', '--save_best', action='store_true')
    parser.add_argument('-sf', '--save_final', action='store_true')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-lrd', '--lr_decay', type=float, default=0.99)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-hid', '--hidden_size', type=int, default=48)
    parser.add_argument('-edd', '--edge_dropout', type=float, default=0.3)
    parser.add_argument('-mdd', '--model_dropout', type=float, default=0.2)
    parser.add_argument('-es', '--early_stopping', type=int, default=0)
    parser.add_argument('-c', '--cpu', type=int, default=1)
    parser.add_argument('-or', '--output_relu', default='none', choices=['relu', 'leaky_relu', 'none'])
    parser.add_argument('-i', '--inductive', default='trans', choices=['normal', 'opt', 'trans'])
    parser.add_argument('-sa', '--subpath_activation', action='store_true')
    parser.add_argument('-ci', '--cell_init', default='none', choices=['none', 'pca'])
    parser.add_argument('-bas', '--batch_seperation', action='store_true')
    parser.add_argument('-pwpath', '--pathway_path', default='./data/h.all.v7.4')
    parser.add_argument('-seed', '--random_seed', type=int, default=777)  #random.randint(0, 2147483647))
    parser.add_argument('-ws', '--weighted_sum', action='store_true')
    parser.add_argument('-samp', '--sampling', action='store_true')
    parser.add_argument('-ns', '--node_sampling_rate', type=float, default=0.5)
    parser.add_argument('-prep', '--preprocessing', default='none', choices=['none', 'feature_selection', 'pca'])

    args = parser.parse_args()

    # For test only (low gpu memory setting; to reproduce competition result need >20G memory - v100)
    if True:
        args.preprocessing = 'feature_selection'
        args.no_pathway = True
        args.sampling = True
        args.batch_size = 10000
        args.epoch = 10
    elif args.subtask == 'openproblems_bmmc_multiome_phase2_mod2':
        args.preprocessing = 'feature_selection'
    elif args.subtask in ['openproblems_bmmc_cite_phase2_mod2', 'openproblems_bmmc_multiome_phase2_rna']:
        args.sampling = True
        args.edge_dropout = 0

    # Regular settings
    if args.subtask.find('rna') == -1:
        args.no_pathway = True
    if args.sampling:
        args.no_pathway = True

    set_seed(args.random_seed)
    torch.set_num_threads(args.cpu)

    pipeline(**vars(args))
""" To reproduce scMoGCN on other samples, please refer to command lines belows:
GEX to ADT:
python scmogcn.py --subtask oopenproblems_bmmc_cite_phase2_rna --device cuda

ADT to GEX:
python scmogcn.py --subtask openproblems_bmmc_cite_phase2_mod2 --device cuda

GEX to ATAC:
python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_rna --device cuda

ATAC to GEX:
python scmogcn.py --subtask openproblems_bmmc_multiome_phase2_mod2 --device cuda
"""
