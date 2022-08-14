import argparse
import random

import numpy as np
import torch

from dance.datasets.singlemodality import ImputationDataset
from dance.modules.single_modality.imputation.scgnn import scGNN

parser = argparse.ArgumentParser(description='Main entrance of scGNN')

parser.add_argument("--data_dir", type=str, default='data', help='data directory')
parser.add_argument("--save_dir", type=str, default='result', help='save directory')
parser.add_argument("--filetype", type=str, default='h5', choices=['csv', 'gz', 'h5'],
                    help='data file type, csv, csv.gz, or h5')
parser.add_argument("--train_dataset", default='mouse_brain_data', type=str, help="dataset id")

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 12800)')
parser.add_argument('--Regu_epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train in Feature Autoencoder initially (default: 500)')
parser.add_argument('--EM_epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train Feature Autoencoder in iteration EM (default: 200)')
parser.add_argument(
    '--EM_iteration',
    type=int,
    default=10,
    metavar='N',  #10
    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--quickmode', action='store_true', default=True,
                    help='whether use quickmode, skip Cluster Autoencoder (default: no quickmode)')
parser.add_argument('--cluster_epochs', type=int, default=200, metavar='N',
                    help='number of epochs in Cluster Autoencoder training (default: 200)')
parser.add_argument("--gpu", type=int, default=1, help="GPU id, -1 for cpu")
parser.add_argument('--random_seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--regulized_type', type=str, default='noregu',
                    help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--reduction', type=str, default='sum', help='reduction type: mean/sum, default(sum)')
parser.add_argument('--model', type=str, default='AE', help='VAE/AE (default: AE)')
parser.add_argument('--gammaPara', type=float, default=0.1, help='regulized intensity (default: 0.1)')
parser.add_argument('--alphaRegularizePara', type=float, default=0.9, help='regulized parameter (default: 0.9)')

# Build cell graph
parser.add_argument('--k', type=int, default=10, help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn_distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument(
    '--prunetype', type=str, default='KNNgraphStatsSingleThread',
    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStatsSingleThread)')

# Debug related
parser.add_argument('--precisionModel', type=str, default='Float',
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--coresUsage', type=str, default='1', help='how many cores used: all/1/... (default:1)')
parser.add_argument('--outputDir', type=str, default='outputDir/', help='save npy results in directory')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--saveinternal', action='store_true', default=False,
                    help='whether save internal interation results or not')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
parser.add_argument('--nonsparseMode', action='store_true', default=False,
                    help='SparseMode for running for huge dataset')

# LTMG related
parser.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                    help='expression File after ltmg in csv')
parser.add_argument(
    '--ltmgFile', type=str, default='LTMG_sparse.mtx',
    help='expression File in csv. (default:LTMG_sparse.mtx for sparse mode/ ltmg.csv for nonsparse mode) ')

# Clustering related
parser.add_argument('--useGAEembedding', action='store_true', default=True,
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False,
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--n_clusters', default=20, type=int, help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument(
    '--clustering_method', type=str, default='Louvain', help=
    'Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB'
)
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)')
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')

# imputation related
parser.add_argument('--EMregulized_type', type=str, default='Celltype',
                    help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR/Celltype')
parser.add_argument('--gammaImputePara', type=float, default=0.0, help='regulized parameter (default: 0.0)')
parser.add_argument('--graphImputePara', type=float, default=0.3, help='graph parameter (default: 0.3)')
parser.add_argument('--celltypeImputePara', type=float, default=0.1, help='celltype parameter (default: 0.1)')
parser.add_argument('--L1Para', type=float, default=1.0, help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0, help='L2 regulized parameter (default: 0.001)')
parser.add_argument('--EMreguTag', action='store_true', default=False, help='whether regu in EM process')
parser.add_argument('--sparseImputation', type=str, default='nonsparse',
                    help='whether use sparse in imputation: sparse/nonsparse (default: nonsparse)')

# dealing with zeros in imputation results
parser.add_argument('--zerofillFlag', action='store_true', default=False,
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--noPostprocessingTag', action='store_false', default=True,
                    help='whether postprocess imputated results, default: (True)')
parser.add_argument('--postThreshold', type=float, default=0.01,
                    help='Threshold to force expression as 0, default:(0.01)')

# Converge related
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge_type', type=str, default='celltype',
                    help='type of converge condition: celltype/graph/both/either (default: celltype) ')
parser.add_argument('--converge_graphratio', type=float, default=0.01,
                    help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge_celltyperatio', type=float, default=0.99,
                    help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

# GAE related
parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
parser.add_argument(
    '--GAEepochs',
    type=int,
    default=2,  #200
    help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')

params = parser.parse_args()

params.quickmode = True
params.cuda = True if params.gpu != -1 and torch.cuda.is_available() else False
params.sparseMode = not params.nonsparseMode

params.device = torch.device(f'cuda:{params.gpu}' if params.cuda else "cpu")
print('Using device:' + str(params.device))

if not params.coresUsage == 'all':
    torch.set_num_threads(int(params.coresUsage))
params.kwargs = {'num_workers': 1, 'pin_memory': True} if params.cuda else {}
# print(params)

dataloader = ImputationDataset(random_seed=params.random_seed, gpu=params.gpu, data_dir=params.data_dir,
                               train_dataset=params.train_dataset, filetype=params.filetype)
dataloader.download_all_data()
# dataloader.download_pretrained_data()

dataloader.load_data(params, model='scGNN')
dl_params = dataloader.params

random.seed(params.random_seed)
np.random.seed(params.random_seed)
torch.manual_seed(params.random_seed)
torch.cuda.manual_seed(params.random_seed)

model = scGNN(dl_params.adata, dl_params.genelist, dl_params.celllist, params, dropout=params.GAEdropout,
              GAEepochs=params.GAEepochs, Regu_epochs=params.Regu_epochs, EM_epochs=params.EM_epochs,
              cluster_epochs=params.cluster_epochs, debugMode=params.debugMode)

model.fit(dl_params.train_data)
recon = model.predict(dl_params.test_data)
mse_cells, mse_genes = model.score(dl_params.test_data, recon, dl_params.test_idx, metric='MSE')
score = mse_cells.mean(axis=0).item()
print("MSE: %.4f" % score)
"""To reproduce GraphSCI benchmarks, please refer to command lines belows:

Mouse Brain:
$ python scgnn.py --train_dataset mouse_brain_data

Mouse Embryo:
$ python scgnn.py --train_dataset mouse_embryo_data
"""
