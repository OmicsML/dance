import copy
import math
import multiprocessing
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class conGraphConvolutionlayer(Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class conGCN(nn.Module):

    def __init__(
        self,
        nfeat,
        nhid,
        common_hid_layers_num,
        fcnn_hid_layers_num,
        dropout,
        nout1,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True

        ## The beginning layer
        self.gc_in_exp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)
        self.gc_in_sp = conGraphConvolutionlayer(nfeat, nhid)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        ## common_hid_layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec('self.cgc{}_exp = conGraphConvolutionlayer(nhid, nhid)'.format(i + 1))
                exec('self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)'.format(i + 1))
                exec('self.cgc{}_sp = conGraphConvolutionlayer(nhid, nhid)'.format(i + 1))
                exec('self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)'.format(i + 1))

        ## FCNN layers
        self.gc_out11 = nn.Linear(2 * nhid, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec('self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)'.format(i + 1))
                exec('self.bn_out11{} = nn.BatchNorm1d(nhid)'.format(i + 1))
        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)

    def forward(self, x, adjs):

        self.x = x

        ## input layer
        self.x_exp = self.gc_in_exp(self.x, adjs[0])
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        self.x_sp = self.gc_in_sp(self.x, adjs[1])
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## common layers
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("self.x_exp = self.cgc{}_exp(self.x_exp, adjs[0])".format(i + 1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i + 1))
                self.x_exp = F.elu(self.x_exp)
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
                exec("self.x_sp = self.cgc{}_sp(self.x_sp, adjs[1])".format(i + 1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i + 1))
                self.x_sp = F.elu(self.x_sp)
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        ## FCNN layers
        self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)
        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i + 1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i + 1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        self.x1 = self.gc_out12(self.x1)

        gc_list = {}
        gc_list['gc_in_exp'] = self.gc_in_exp
        gc_list['gc_in_sp'] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i + 1, i + 1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i + 1, i + 1))
        gc_list['gc_out11'] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i + 1, i + 1))
        gc_list['gc_out12'] = self.gc_out12

        return F.log_softmax(self.x1, dim=1), gc_list


def conGCN_train(model, train_valid_len, test_len, feature, adjs, label, epoch_n, loss_fn, optimizer,
                 train_valid_ratio=0.9, scheduler=None, early_stopping_patience=5, clip_grad_max_norm=1,
                 load_test_groundtruth=False, print_epoch_step=1, cpu_num=-1, GCN_device='CPU'):

    if GCN_device == 'CPU':
        device = torch.device("cpu")
        print('Use CPU as device.')
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Use GPU as device.')
        else:
            device = torch.device("cpu")
            print('Use CPU as device.')

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)

    model = model.to(device)
    adjs = [adj.to(device) for adj in adjs]
    feature = feature.to(device)
    label = label.to(device)

    time_open = time.time()

    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)

    best_val = np.inf
    clip = 0
    loss = []
    para_list = []
    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        optimizer.zero_grad()
        output1, paras = model(feature.float(), adjs)

        loss_train1 = loss_fn(output1[list(np.array(train_idx) + test_len)],
                              label[list(np.array(train_idx) + test_len)].float())
        loss_val1 = loss_fn(output1[list(np.array(valid_idx) + test_len)],
                            label[list(np.array(valid_idx) + test_len)].float())
        if load_test_groundtruth == True:
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])

        if epoch % print_epoch_step == 0:
            print("******************************************")
            print("Epoch {}/{}".format(epoch + 1, epoch_n), 'loss_train: {:.4f}'.format(loss_train1.item()),
                  'loss_val: {:.4f}'.format(loss_val1.item()), end='\t')
            if load_test_groundtruth == True:
                print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
            print('time: {:.4f}s'.format(time.time() - time_open))
        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])

        if early_stopping_patience > 0:
            if torch.round(loss_val1, decimals=4) < best_val:
                best_val = torch.round(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None

        loss_train1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=loss_val1)

    print("***********************Final Loss***********************")
    print("Epoch {}/{}".format(epoch + 1, epoch_n), 'loss_train: {:.4f}'.format(loss_train1.item()),
          'loss_val: {:.4f}'.format(loss_val1.item()), end='\t')
    if load_test_groundtruth == True:
        print("Test loss= {:.4f}".format(loss_test1.item()), end='\t')
    print('time: {:.4f}s'.format(time.time() - time_open))

    torch.cuda.empty_cache()

    return output1.cpu(), loss, model.cpu()


import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


def run_STdGCN(paths, find_marker_genes_paras, pseudo_spot_simulation_paras, data_normalization_paras,
               integration_for_adj_paras, inter_exp_adj_paras, spatial_adj_paras, real_intra_exp_adj_paras,
               pseudo_intra_exp_adj_paras, integration_for_feature_paras, GCN_paras, load_test_groundtruth=False,
               use_marker_genes=True, external_genes=False, generate_new_pseudo_spots=True, fraction_pie_plot=False,
               cell_type_distribution_plot=True, n_jobs=-1, GCN_device='CPU'):

    sc_path = paths['sc_path']
    ST_path = paths['ST_path']
    output_path = paths['output_path']

    sc_adata = sc.read_csv(sc_path + "/sc_data.tsv", delimiter='\t')
    sc_label = pd.read_table(sc_path + "/sc_label.tsv", sep='\t', header=0, index_col=0, encoding="utf-8")
    sc_label.columns = ['cell_type']
    sc_adata.obs['cell_type'] = sc_label['cell_type'].values

    cell_type_num = len(sc_adata.obs['cell_type'].unique())
    cell_types = sc_adata.obs['cell_type'].unique()

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
    sc_adata.obs['cell_type_idx'] = celltype_idx
    sc_adata.obs['cell_type'].value_counts()

    if use_marker_genes == True:
        if external_genes == True:
            with open(sc_path + "/marker_genes.tsv") as f:
                selected_genes = [line.rstrip('\n') for line in f]
        else:
            selected_genes, cell_type_marker_genes = find_marker_genes(
                sc_adata, preprocess=find_marker_genes_paras['preprocess'],
                highly_variable_genes=find_marker_genes_paras['highly_variable_genes'],
                PCA_components=find_marker_genes_paras['PCA_components'],
                filter_wilcoxon_marker_genes=find_marker_genes_paras['filter_wilcoxon_marker_genes'],
                marker_gene_method=find_marker_genes_paras['marker_gene_method'],
                pvals_adj_threshold=find_marker_genes_paras['pvals_adj_threshold'],
                log_fold_change_threshold=find_marker_genes_paras['log_fold_change_threshold'],
                min_within_group_fraction_threshold=find_marker_genes_paras['min_within_group_fraction_threshold'],
                max_between_group_fraction_threshold=find_marker_genes_paras['max_between_group_fraction_threshold'],
                top_gene_per_type=find_marker_genes_paras['top_gene_per_type'])
            with open(output_path + "/marker_genes.tsv", 'w') as f:
                for gene in selected_genes:
                    f.write(str(gene) + '\n')

    print("{} genes have been selected as marker genes.".format(len(selected_genes)))

    if generate_new_pseudo_spots == True:
        pseudo_adata = pseudo_spot_generation(
            sc_adata, idx_to_word_celltype, spot_num=pseudo_spot_simulation_paras['spot_num'],
            min_cell_number_in_spot=pseudo_spot_simulation_paras['min_cell_num_in_spot'],
            max_cell_number_in_spot=pseudo_spot_simulation_paras['max_cell_num_in_spot'],
            max_cell_types_in_spot=pseudo_spot_simulation_paras['max_cell_types_in_spot'],
            generation_method=pseudo_spot_simulation_paras['generation_method'], n_jobs=n_jobs)
        data_file = open(output_path + '/pseudo_ST.pkl', 'wb')
        pickle.dump(pseudo_adata, data_file)
        data_file.close()
    else:
        data_file = open(output_path + '/pseudo_ST.pkl', 'rb')
        pseudo_adata = pickle.load(data_file)
        data_file.close()

    ST_adata = sc.read_csv(ST_path + "/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path + "/coordinates.csv", sep=',', header=0, index_col=0, encoding="utf-8")
    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']
    if load_test_groundtruth == True:
        ST_groundtruth = pd.read_table(ST_path + "/ST_ground_truth.tsv", sep='\t', header=0, index_col=0,
                                       encoding="utf-8")
        for i in cell_types:
            ST_adata.obs[i] = ST_groundtruth[i]

    ST_genes = ST_adata.var.index.values
    pseudo_genes = pseudo_adata.var.index.values
    common_genes = set(ST_genes).intersection(set(pseudo_genes))
    ST_adata_filter = ST_adata[:, list(common_genes)]
    pseudo_adata_filter = pseudo_adata[:, list(common_genes)]

    ST_adata_filter_norm = ST_preprocess(
        ST_adata_filter,
        normalize=data_normalization_paras['normalize'],
        log=data_normalization_paras['log'],
        scale=data_normalization_paras['scale'],
    )[:, selected_genes]

    try:
        try:
            ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
        except:
            ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']
    except:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', [0] * ST_adata_filter_norm.obs.shape[0])
    for i in cell_types:
        try:
            ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
        except:
            ST_adata_filter_norm.obs[i] = [0] * ST_adata_filter_norm.obs.shape[0]
    try:
        ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types] > 0).sum(axis=1)
    except:
        ST_adata_filter_norm.obs['cell_type_num'] = [0] * ST_adata_filter_norm.obs.shape[0]

    pseudo_adata_norm = ST_preprocess(
        pseudo_adata_filter,
        normalize=data_normalization_paras['normalize'],
        log=data_normalization_paras['log'],
        scale=data_normalization_paras['scale'],
    )[:, selected_genes]

    pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types] > 0).sum(axis=1)

    ST_integration = data_integration(
        ST_adata_filter_norm, pseudo_adata_norm, batch_removal_method=integration_for_adj_paras['batch_removal_method'],
        dim=min(integration_for_adj_paras['dim'], int(ST_adata_filter_norm.shape[1] / 2)),
        dimensionality_reduction_method=integration_for_adj_paras['dimensionality_reduction_method'],
        scale=integration_for_adj_paras['scale'], cpu_num=n_jobs, AE_device=GCN_device)

    A_inter_exp = inter_adj(
        ST_integration,
        find_neighbor_method=inter_exp_adj_paras['find_neighbor_method'],
        dist_method=inter_exp_adj_paras['dist_method'],
        corr_dist_neighbors=inter_exp_adj_paras['corr_dist_neighbors'],
    )

    A_intra_space = intra_dist_adj(
        ST_adata_filter_norm,
        link_method=spatial_adj_paras['link_method'],
        space_dist_threshold=spatial_adj_paras['space_dist_threshold'],
    )

    A_real_intra_exp = intra_exp_adj(
        ST_adata_filter_norm,
        find_neighbor_method=real_intra_exp_adj_paras['find_neighbor_method'],
        dist_method=real_intra_exp_adj_paras['dist_method'],
        PCA_dimensionality_reduction=real_intra_exp_adj_paras['PCA_dimensionality_reduction'],
        corr_dist_neighbors=real_intra_exp_adj_paras['corr_dist_neighbors'],
    )

    A_pseudo_intra_exp = intra_exp_adj(
        pseudo_adata_norm,
        find_neighbor_method=pseudo_intra_exp_adj_paras['find_neighbor_method'],
        dist_method=pseudo_intra_exp_adj_paras['dist_method'],
        PCA_dimensionality_reduction=pseudo_intra_exp_adj_paras['PCA_dimensionality_reduction'],
        corr_dist_neighbors=pseudo_intra_exp_adj_paras['corr_dist_neighbors'],
    )

    real_num = ST_adata_filter.shape[0]
    pseudo_num = pseudo_adata.shape[0]

    adj_inter_exp = A_inter_exp.values
    adj_pseudo_intra_exp = A_intra_transfer(A_pseudo_intra_exp, 'pseudo', real_num, pseudo_num)
    adj_real_intra_exp = A_intra_transfer(A_real_intra_exp, 'real', real_num, pseudo_num)
    adj_intra_space = A_intra_transfer(A_intra_space, 'real', real_num, pseudo_num)

    adj_alpha = 1
    adj_beta = 1
    diag_power = 20
    adj_balance = (1 + adj_alpha + adj_beta) * diag_power
    adj_exp = torch.tensor(adj_inter_exp + adj_alpha * adj_pseudo_intra_exp +
                           adj_beta * adj_real_intra_exp) / adj_balance + torch.eye(adj_inter_exp.shape[0])
    adj_sp = torch.tensor(adj_intra_space) / diag_power + torch.eye(adj_intra_space.shape[0])

    norm = True
    if (norm == True):
        adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))
        adj_sp = torch.tensor(adj_normalize(adj_sp, symmetry=True))

    ST_integration_batch_removed = data_integration(
        ST_adata_filter_norm, pseudo_adata_norm,
        batch_removal_method=integration_for_feature_paras['batch_removal_method'],
        dim=min(int(ST_adata_filter_norm.shape[1] * 1 / 2), integration_for_feature_paras['dim']),
        dimensionality_reduction_method=integration_for_feature_paras['dimensionality_reduction_method'],
        scale=integration_for_feature_paras['scale'], cpu_num=n_jobs, AE_device=GCN_device)
    feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)

    input_layer = feature.shape[1]
    hidden_layer = min(int(ST_adata_filter_norm.shape[1] * 1 / 2), GCN_paras['dim'])
    output_layer1 = len(word_to_idx_celltype)
    epoch_n = GCN_paras['epoch_n']
    common_hid_layers_num = GCN_paras['common_hid_layers_num']
    fcnn_hid_layers_num = GCN_paras['fcnn_hid_layers_num']
    dropout = GCN_paras['dropout']
    learning_rate_SGD = GCN_paras['learning_rate_SGD']
    weight_decay_SGD = GCN_paras['weight_decay_SGD']
    momentum = GCN_paras['momentum']
    dampening = GCN_paras['dampening']
    nesterov = GCN_paras['nesterov']
    early_stopping_patience = GCN_paras['early_stopping_patience']
    clip_grad_max_norm = GCN_paras['clip_grad_max_norm']
    LambdaLR_scheduler_coefficient = 0.997
    ReduceLROnPlateau_factor = 0.1
    ReduceLROnPlateau_patience = 5
    scheduler = 'scheduler_ReduceLROnPlateau'
    print_epoch_step = GCN_paras['print_loss_epoch_step']
    cpu_num = n_jobs

    model = conGCN(nfeat=input_layer, nhid=hidden_layer, common_hid_layers_num=common_hid_layers_num,
                   fcnn_hid_layers_num=fcnn_hid_layers_num, dropout=dropout, nout1=output_layer1)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_SGD, momentum=momentum,
                                weight_decay=weight_decay_SGD, dampening=dampening, nesterov=nesterov)

    scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: LambdaLR_scheduler_coefficient**epoch)
    scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                             factor=ReduceLROnPlateau_factor,
                                                                             patience=ReduceLROnPlateau_patience,
                                                                             threshold=0.0001, threshold_mode='rel',
                                                                             cooldown=0, min_lr=0)
    if scheduler == 'scheduler_LambdaLR':
        scheduler = scheduler_LambdaLR
    elif scheduler == 'scheduler_ReduceLROnPlateau':
        scheduler = scheduler_ReduceLROnPlateau
    else:
        scheduler = None

    loss_fn1 = nn.KLDivLoss(reduction='mean')

    train_valid_len = pseudo_adata.shape[0]
    test_len = ST_adata_filter.shape[0]

    table1 = ST_adata_filter_norm.obs.copy()
    label1 = table1[pseudo_adata.obs.iloc[:, :-1].columns].append(pseudo_adata.obs.iloc[:, :-1])
    label1 = torch.tensor(label1.values)

    adjs = [adj_exp.float(), adj_sp.float()]

    output1, loss, trained_model = conGCN_train(
        model=model, train_valid_len=train_valid_len, train_valid_ratio=0.9, test_len=test_len, feature=feature,
        adjs=adjs, label=label1, epoch_n=epoch_n, loss_fn=loss_fn1, optimizer=optimizer, scheduler=scheduler,
        early_stopping_patience=early_stopping_patience, clip_grad_max_norm=clip_grad_max_norm,
        load_test_groundtruth=load_test_groundtruth, print_epoch_step=print_epoch_step, cpu_num=cpu_num,
        GCN_device=GCN_device)

    loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(loss_table.index, loss_table['train'], label='train')
    ax.plot(loss_table.index, loss_table['valid'], label='valid')
    if load_test_groundtruth == True:
        ax.plot(loss_table.index, loss_table['test'], label='test')
    ax.set_xlabel('Epoch', fontsize=20)
    ax.set_ylabel('Loss', fontsize=20)
    ax.set_title('Loss function curve', fontsize=20)
    ax.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path + '/Loss_function.jpg', dpi=300)
    plt.close('all')

    predict_table = pd.DataFrame(
        np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index,
        columns=pseudo_adata_norm.obs.columns[:-2])
    predict_table.to_csv(output_path + '/predict_result.csv', index=True, header=True)

    torch.save(trained_model, output_path + '/model_parameters')

    pred_use = np.round_(output1.exp().detach()[:test_len], decimals=4)
    cell_type_list = cell_types
    coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]

    if fraction_pie_plot == True:
        plot_frac_results(pred_use, cell_type_list, coordinates, point_size=300, size_coefficient=0.0009,
                          file_name=output_path + '/predict_results_pie_plot.jpg', if_show=False)

    if cell_type_distribution_plot == True:
        plot_scatter_by_type(pred_use, cell_type_list, coordinates, point_size=300, file_path=output_path,
                             if_show=False)

    ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())

    torch.cuda.empty_cache()

    return ST_adata_filter_norm


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import DistanceMetric, KDTree, NearestNeighbors


def find_mutual_nn(
    data1,
    data2,
    dist_method,
    k1,
    k2,
):
    if dist_method == 'cosine':
        cos_sim1 = cosine_similarity(data1, data2)
        cos_sim2 = cosine_similarity(data2, data1)
        k_index_1 = torch.topk(torch.tensor(cos_sim2), k=k2, dim=1)[1]
        k_index_2 = torch.topk(torch.tensor(cos_sim1), k=k1, dim=1)[1]
    else:
        dist = DistanceMetric.get_metric(dist_method)
        k_index_1 = KDTree(data1, metric=dist).query(data2, k=k2, return_distance=False)
        k_index_2 = KDTree(data2, metric=dist).query(data1, k=k1, return_distance=False)
    mutual_1 = []
    mutual_2 = []
    mutual = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
                mutual.append([index_1, index_2])
    return mutual


def inter_adj(
    ST_integration,
    find_neighbor_method='MNN',
    dist_method='euclidean',
    corr_dist_neighbors=20,
):

    if find_neighbor_method == 'KNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        if dist_method == 'cosine':
            cos_sim = cosine_similarity(data1, data2)
            k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
        else:
            dist = DistanceMetric.get_metric(dist_method)
            k_index = KDTree(data2, metric=dist).query(data1, k=corr_dist_neighbors, return_distance=False)
        A_exp = np.zeros((ST_integration.shape[0], ST_integration.shape[0]), dtype=float)
        for i in range(k_index.shape[0]):
            for j in k_index[i]:
                A_exp[i, j + real_num] = 1
                A_exp[j + real_num, i] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)

    elif find_neighbor_method == 'MNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        mut = find_mutual_nn(data2, data1, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
        mut = pd.DataFrame(mut, columns=['pseudo', 'real'])
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        A_exp = np.zeros((real_num + pseudo_num, real_num + pseudo_num), dtype=float)
        for i in mut.index:
            A_exp[mut.loc[i, 'real'], mut.loc[i, 'pseudo'] + real_num] = 1
            A_exp[mut.loc[i, 'pseudo'] + real_num, mut.loc[i, 'real']] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)

    return A_exp


def intra_dist_adj(ST_exp, link_method='soft', space_dist_neighbors=27, space_dist_threshold=None):

    knn = NearestNeighbors(n_neighbors=space_dist_neighbors, metric='minkowski')

    knn.fit(ST_exp.obs[['coor_X', 'coor_Y']])
    dist, ind = knn.kneighbors()

    if link_method == 'hard':
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i, j] < space_dist_threshold:
                        A_space[i, ind[i, j]] = 1
                        A_space[ind[i, j], i] = 1
                else:
                    A_space[i, ind[i, j]] = 1
                    A_space[ind[i, j], i] = 1
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i, j] < space_dist_threshold:
                        A_space[i, ind[i, j]] = 1 / dist[i, j]
                        A_space[ind[i, j], i] = 1 / dist[i, j]
                else:
                    A_space[i, ind[i, j]] = 1 / dist[i, j]
                    A_space[ind[i, j], i] = 1 / dist[i, j]
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_space


def intra_exp_adj(
    adata,
    find_neighbor_method='KNN',
    dist_method='euclidean',
    PCA_dimensionality_reduction=True,
    dim=50,
    corr_dist_neighbors=10,
):

    ST_exp = adata.copy()

    sc.pp.scale(ST_exp, max_value=None, zero_center=True)
    if PCA_dimensionality_reduction == True:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1
                        A_exp[j, i] = 1
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        input_data = ST_exp.X
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1
                        A_exp[j, i] = 1
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_exp


def A_intra_transfer(data, data_type, real_num, pseudo_num):

    adj = np.zeros((real_num + pseudo_num, real_num + pseudo_num), dtype=float)
    if data_type == 'real':
        adj[:real_num, :real_num] = data
    elif data_type == 'pseudo':
        adj[real_num:, real_num:] = data

    return adj


def adj_normalize(mx, symmetry=True):

    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    if symmetry == True:
        r_mat_inv = sp.diags(np.sqrt(r_inv))
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)

    return mx.todense()


import multiprocessing
import time

import scanpy as sc
import torch
import torch.nn as nn


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class autoencoder(nn.Module):

    def __init__(self, x_size, hidden_size, embedding_size, p_drop=0):
        super().__init__()

        self.encoder = nn.Sequential(full_block(x_size, hidden_size, p_drop),
                                     full_block(hidden_size, embedding_size, p_drop))

        self.decoder = nn.Sequential(full_block(embedding_size, hidden_size, p_drop),
                                     full_block(hidden_size, x_size, p_drop))

    def forward(self, x):

        en = self.encoder(x)
        de = self.decoder(en)

        return en, de, [self.encoder, self.decoder]


def auto_train(model, epoch_n, loss_fn, optimizer, data, cpu_num=-1, device='GPU'):

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)

    if device == 'GPU':
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()

    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        train_cost = 0

        optimizer.zero_grad()
        en, de, _ = model(data)

        loss = loss_fn(de, data)

        loss.backward()
        optimizer.step()

    torch.cuda.empty_cache()

    return en.cpu()


import multiprocessing
import random

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.decomposition import NMF
from tqdm.notebook import tqdm


def ST_preprocess(ST_exp, normalize=True, log=True, highly_variable_genes=False, regress_out=False, scale=False,
                  scale_max_value=None, scale_zero_center=True, hvg_min_mean=0.0125, hvg_max_mean=3, hvg_min_disp=0.5,
                  highly_variable_gene_num=None):

    adata = ST_exp.copy()

    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log == True:
        sc.pp.log1p(adata)

    adata.layers['scale.data'] = adata.X.copy()

    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(
            adata,
            min_mean=hvg_min_mean,
            max_mean=hvg_max_mean,
            min_disp=hvg_min_disp,
            n_top_genes=highly_variable_gene_num,
        )
        adata = adata[:, adata.var.highly_variable]

    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)

    return adata


def find_marker_genes(
    sc_exp,
    preprocess=True,
    highly_variable_genes=True,
    regress_out=False,
    scale=False,
    PCA_components=50,
    marker_gene_method='wilcoxon',
    filter_wilcoxon_marker_genes=True,
    top_gene_per_type=20,
    pvals_adj_threshold=0.10,
    log_fold_change_threshold=1,
    min_within_group_fraction_threshold=0.7,
    max_between_group_fraction_threshold=0.3,
):

    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(
            sc_exp.copy(),
            normalize=True,
            log=True,
            highly_variable_genes=highly_variable_genes,
            regress_out=regress_out,
            scale=scale,
        )
    else:
        sc_adata_marker_gene = sc_exp.copy()

    sc.tl.pca(sc_adata_marker_gene, n_comps=PCA_components, svd_solver='arpack', random_state=None)

    layer = 'scale.data'
    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True,
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)

    if marker_gene_method == 'wilcoxon':
        if filter_wilcoxon_marker_genes == True:
            gene_dict = {}
            gene_list = []
            for name in sc_adata_marker_gene.obs['cell_type'].unique():
                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name,
                                                   key=marker_gene_method).sort_values('pvals_adj')
                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]
                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]
                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]
                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]
                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list({item for sublist in gene_table.values.tolist() for item in sublist})
    else:
        print("marker_gene_method should be 'logreg' or 'wilcoxon'")

    return gene_list, gene_dict


def generate_a_spot(
    sc_exp,
    min_cell_number_in_spot,
    max_cell_number_in_spot,
    max_cell_types_in_spot,
    generation_method,
):

    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)

        while (True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]

        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])

        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')


def pseudo_spot_generation(sc_exp, idx_to_word_celltype, spot_num, min_cell_number_in_spot, max_cell_number_in_spot,
                           max_cell_types_in_spot, generation_method, n_jobs=-1):

    cell_type_num = len(sc_exp.obs['cell_type'].unique())

    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method)
            for i in range(spot_num)]
    generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))

    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:, :].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i] / pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)

    return pseudo_spots


def data_integration(real, pseudo, batch_removal_method="combat", dimensionality_reduction_method='PCA', dim=50,
                     scale=True, autoencoder_epoches=2000, autoencoder_LR=1e-3, autoencoder_drop=0, cpu_num=-1,
                     AE_device='GPU'):

    if batch_removal_method == 'mnn':
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True,
                                         var_subset=None)
        adata = mnn[0]
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table = table.iloc[pseudo.shape[0]:, :].append(table.iloc[:pseudo.shape[0], :])
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    elif batch_removal_method == 'scanorama':
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred=dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = table1.append(table2)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    elif batch_removal_method == 'combat':
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index=aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index=bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    else:
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index=aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index=bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size) / 2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size,
                               p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction='mean')
            embedding = auto_train(model=nets, epoch_n=autoencoder_epoches, loss_fn=loss_ae, optimizer=optimizer_ae,
                                   data=data, cpu_num=cpu_num, device=AE_device).detach().numpy()
            if scale == True:
                embedding = (embedding - embedding.mean(axis=0)) / embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf - nmf.mean(axis=0)) / nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real'] * real.shape[0] + ['pseudo'] * pseudo.shape[0])

    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist() + pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num',
                 real.obs['cell_type_num'].values.tolist() + pseudo.obs['cell_type_num'].values.tolist())

    return table


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm


def draw_pie(dist, xpos, ypos, size, colors, ax):

    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    i = 0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=30)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, c=colors[i], edgecolors='none')
        i += 1

    return ax


def plot_frac_results(predict, cell_type_list, coordinates, file_name=None, point_size=1000, size_coefficient=0.0009,
                      if_show=True, color_dict=None):

    coordinates.columns = ['coor_X', 'coor_Y']
    labels = cell_type_list
    if color_dict != None:
        colors = []
        for i in cell_type_list:
            colors.append(color_dict[i])
    else:
        if len(labels) <= 10:
            colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
        else:
            import matplotlib
            color = plt.get_cmap('rainbow', len(labels))
            colors = []
            for x in color([range(len(labels))][0]):
                colors.append(matplotlib.colors.to_hex(x, keep_alpha=False))

    str_len = 0
    for item in cell_type_list:
        str_len = max(str_len, len(item))
    extend_region = str_len / 15 + 3

    fig, ax = plt.subplots(figsize=(len(coordinates['coor_X'].unique()) * point_size * size_coefficient + extend_region,
                                    len(coordinates['coor_Y'].unique()) * point_size * size_coefficient))

    for i in tqdm(range(predict.shape[0]), desc="Plotting pie plots:"):
        ax = draw_pie(predict[i], coordinates['coor_X'].values[i], coordinates['coor_Y'].values[i], size=point_size,
                      ax=ax, colors=colors)

    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(len(colors))]
    fontsize = max(predict.shape[0] / 100, 10)
    fontsize = min(fontsize, 30)
    ax.legend(handles=patches, fontsize=fontsize, bbox_to_anchor=(1, 1), loc="upper left")
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if file_name != None:
        plt.savefig(
            file_name,
            dpi=300,
            #bbox_inches='tight'
        )
    if if_show == True:
        plt.show()
    plt.close('all')


def plot_scatter_by_type(predict, cell_type_list, coordinates, point_size=400, size_coefficient=0.0009, file_path=None,
                         if_show=True):

    coordinates.columns = ['coor_X', 'coor_Y']

    for i in tqdm(range(len(cell_type_list)), desc="Plotting cell type scatter plot:"):

        fig, ax = plt.subplots(figsize=(len(coordinates['coor_X'].unique()) * point_size * size_coefficient + 1,
                                        len(coordinates['coor_Y'].unique()) * point_size * size_coefficient))
        cm = plt.cm.get_cmap('Reds')
        ax = plt.scatter(coordinates['coor_X'], coordinates['coor_Y'], s=point_size, vmin=0, vmax=1, c=predict[:, i],
                         cmap=cm)

        cbar = plt.colorbar(ax, fraction=0.05)
        labelsize = max(predict.shape[0] / 100, 10)
        labelsize = min(labelsize, 30)
        cbar.ax.tick_params(labelsize=labelsize)
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(coordinates['coor_X'].min() - 0.5, coordinates['coor_X'].max() + 0.5)
        plt.ylim(coordinates['coor_Y'].min() - 0.5, coordinates['coor_Y'].max() + 0.5)
        plt.tight_layout()
        if file_path != None:
            name = cell_type_list[i].replace('/', '_')
            plt.savefig(file_path + '/{}.jpg'.format(name), dpi=300, bbox_inches='tight')
        if if_show == True:
            plt.show()
        plt.close('all')
