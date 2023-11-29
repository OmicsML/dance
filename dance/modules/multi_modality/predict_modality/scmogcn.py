"""Official release of scMoGNN method.

Reference
---------
Wen, Hongzhi, et al. "Graph Neural Networks for Multimodal Single-Cell Data Integration." arXiv preprint arXiv:2203.01884 (2022).

"""
import copy
import math
from copy import deepcopy

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dance.utils import SimpleIndexDataset


class ScMoGCNWrapper:
    """ScMoGCN class.

    Parameters
    ----------
    args : argparse.Namespace
        A Namespace object that contains arguments of ScMoGCN. For details of parameters in parser args, please refer to link (parser help document).

    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = ScMoGCN(args).to(args.device)

    def predict(self, graph, idx=None, device='cpu'):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Cell-feature graph contructed from the dataset.
        idx : Iterable[int] optional
            Cell indices for prediction, by default to be None, where all the cells to be predicted.
        device : str optional
            Well to perform predicting, by default to be 'gpu'.

        Returns
        -------
        pred : torch.Tensor
            Predicted target modality features.

        """
        if self.args.device != 'cpu' and device == 'cpu':
            model = copy.deepcopy(self.model)
            model.to('cpu')
            graph = graph.to('cpu')
        else:
            model = self.model
        model.eval()
        with torch.no_grad():
            if idx is None:
                pred = model.forward(graph)
            else:
                pred = model.forward(graph)[idx]
        return pred.to(device)

    def score(self, g, idx, labels, device='cpu'):
        """Score function to get score of prediction.

        Parameters
        ----------
        g : dgl.DGLGraph
            Cell-feature graph contructed from the dataset.
        idx : Iterable[int] optional
            Index of testing cells for scoring.
        labels : torch.Tensor
            Ground truth label of cells, a.k.s target modality features.
        device : str optional
            Well to perform predicting, by default to be 'gpu'.

        Returns
        -------
        loss : float
            RMSE loss of predicted output modality features.

        """
        self.model.eval()
        with torch.no_grad():
            logits = F.relu(self.predict(g, idx, device))
            loss = math.sqrt(F.mse_loss(logits, labels).item())
            return loss

    # TODO: need to modify the logic of validation and test to adapt Inductive learning;
    #  w. test = Transductive learning, w/o = Inductive learning
    def fit(self, g, y, split=None, eval=True, verbose=2, y_test=None, logger=None, sampling=False, eval_interval=1):
        """Fit function for training.

        Parameters
        ----------
        g : dgl.DGLGraph
            Cell-feature graph contructed from the dataset.
        y : torch.Tensor
            Labels of each training cell, a.k.a target modality features.
        split : dictionary optional
            Cell indices for train-test split, needed when eval parameter set to be True.
        eval : bool optional
            Whether to evaluate during training, by default to be True.
        verbose : int optional
            Verbose level, by default to be 2 (i.e. print and logger).
        y_test : torch.Tensor optional
            Labels of each testing cell, needed when eval parameter set to be True.
        logger : file-object optional
            Log file, needed when verbose set to be 2.
        sampling : bool optional
            Whether perform feature and cell sampling, by default to be False.

        Returns
        -------
        None.

        """
        if sampling:
            return self.fit_with_sampling(g, y, split, eval, verbose, y_test, logger)
        kwargs = vars(self.args)
        PREFIX = kwargs['prefix']
        CELL_SIZE = kwargs['CELL_SIZE']
        TRAIN_SIZE = kwargs['TRAIN_SIZE']

        g = g.to(self.args.device)
        y = y.float().to(self.args.device)
        y_test = y_test.float().to(self.args.device) if y_test is not None else None

        if verbose > 1 and logger is None:
            logger = open(f'{kwargs["log_folder"]}/{PREFIX}.log', 'w')
        if verbose > 1:
            logger.write(str(self.model) + '\n')
            logger.flush()

        opt = torch.optim.AdamW(self.model.parameters(), lr=kwargs['learning_rate'],
                                weight_decay=kwargs['weight_decay'])
        criterion = nn.MSELoss()
        val = []
        tr = []
        te = []
        minval = 100
        minvep = -1

        for epoch in range(kwargs['epoch']):
            if verbose > 1:
                logger.write(f'epoch:  {epoch}\n')

            self.model.train()
            logits = self.model(g)
            loss = criterion(logits[split['train']], y[split['train']])
            running_loss = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            torch.cuda.empty_cache()
            tr.append(math.sqrt(running_loss))

            if epoch % eval_interval == 0:
                val.append(self.score(g, split['valid'], y[split['valid']], self.args.device))
                if verbose > 1:
                    logger.write(f'training loss:  {tr[-1]}\n')
                    logger.flush()
                    logger.write(f'validation loss:  {val[-1]}\n')
                    logger.flush()

                if eval:
                    te.append(self.score(g, np.arange(TRAIN_SIZE, CELL_SIZE), y_test, self.args.device))
                    if verbose > 1:
                        logger.write(f'testing loss:  {te[-1]}\n')
                        logger.flush()

                if val[-1] < minval:
                    minval = val[-1]
                    minvep = epoch // eval_interval
                    if kwargs['save_best']:
                        torch.save(self.model, f'{kwargs["model_folder"]}/{PREFIX}.best.pth')
                    best_dict = deepcopy(self.model.state_dict())

                if epoch > 1500 and kwargs['early_stopping'] > 0 and min(val[-kwargs['early_stopping']:]) > minval:
                    if verbose > 1:
                        logger.write('Early stopped.\n')
                    break

                if epoch > 1200:
                    if epoch % 15 == 0:
                        for p in opt.param_groups:
                            p['lr'] *= kwargs['lr_decay']

                if verbose > 0:
                    print('epoch', epoch)
                    print('training: ', tr[-1])
                    print('valid: ', val[-1])
                    if eval:
                        print('testing: ', te[-1])

        if kwargs['save_final']:
            state = {'model': self.model, 'optimizer': opt.state_dict(), 'epoch': epoch - 1}
            torch.save(state, f'{kwargs["model_folder"]}/{PREFIX}.epoch{epoch}.pth')

        if verbose > 1:
            if eval:
                logger.write(
                    f'epoch {minvep} minimal val {minval} with training:  {tr[minvep]} and testing: {te[minvep]}\n')
            else:
                logger.write(f'epoch {minvep} minimal val {minval} with training:  {tr[minvep]}\n')
            logger.close()

        if verbose > 0 and eval:
            print('min testing', min(te), te.index(min(te)))
            print('converged testing', minvep * eval_interval, te[minvep])
        self.model.load_state_dict(best_dict)
        return self.model

    def fit_with_sampling(self, g, y, split=None, eval=True, verbose=2, y_test=None, logger=None, eval_interval=1):
        """Fit function for training with graph sampling.

        Parameters
        ----------
        g : dgl.DGLGraph
            Cell-feature graph contructed from the dataset.
        y : torch.Tensor
            Labels of each training cell, a.k.a target modality features.
        split : dictionary optional
            Cell indices for train-test split, needed when eval parameter set to be True.
        eval : bool optional
            Whether to evaluate during training, by default to be True.
        verbose : int optional
            Verbose level, by default to be 2 (i.e. print and logger).
        y_test : torch.Tensor optional
            Labels of each testing cell, needed when eval parameter set to be True.
        logger : file-object optional
            Log file, needed when verbose set to be 2.

        Returns
        -------
        None.

        """
        kwargs = vars(self.args)
        PREFIX = kwargs['prefix']
        CELL_SIZE = kwargs['CELL_SIZE']
        TRAIN_SIZE = kwargs['TRAIN_SIZE']
        # Make sure the batch size is small enough to cover all splits
        BATCH_SIZE = min(kwargs['batch_size'], min(map(len, split.values())))

        if verbose > 1 and logger is None:
            logger = open(f'{kwargs["log_folder"]}/{PREFIX}.log', 'w')
        if verbose > 1:
            logger.write(str(self.model) + '\n')
            logger.flush()
        g.nodes['cell'].data['label'] = torch.cat([y, y_test], 0)
        g_origin = g
        # g = g.to('cpu')
        g = g.long()
        train_nid = torch.tensor(split['train'])  #.to(self.args.device)
        # sampler = dgl.sampling.PinSAGESampler(g, 'cell')
        # sampler = dgl.dataloading.NeighborSampler([{#('feature', 'pathway', 'feature'):0,
        #                                             ('cell', 'cell2feature', 'feature'):100,
        #                                             ('feature', 'feature2cell', 'cell'):100},
        #                                            {#('feature', 'pathway', 'feature'): 0,
        #                                             ('cell', 'cell2feature', 'feature'): 100,
        #                                             ('feature', 'feature2cell', 'cell'): 100},
        #                                            {#('feature', 'pathway', 'feature'): 0,
        #                                             ('cell', 'cell2feature', 'feature'): 100,
        #                                             ('feature', 'feature2cell', 'cell'): 100},
        #                                            {#('feature', 'pathway', 'feature'): 0,
        #                                             ('cell', 'cell2feature', 'feature'): 100,
        #                                             ('feature', 'feature2cell', 'cell'): 100},],
        #
        #                                             # ('feature', 'pathway' ,'feature'): 5,}]
        #                                           prob = 'weight', output_device='cpu')

        # sampler = dgl.dataloading.SAINTSampler(mode='node', budget=6000, cache=True)
        # dataloader = dgl.dataloading.DataLoader(
        #     g,  # The graph must be on GPU.
        #     {'cell': train_nid},  # train_nid must be on GPU.
        #     sampler,
        #     device = torch.device('cpu'), #torch.device(self.args.device),  # The device argument must be GPU.
        #     num_workers=0,  # Number of workers must be 0.
        #     batch_size=1000,
        #     drop_last=False,
        #     shuffle=True)

        train_dataset = SimpleIndexDataset(split['train'])
        dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        feature_weight = g.in_degrees(etype='cell2feature').float()  # /g.in_degrees(etype='cell2feature').sum()

        opt = torch.optim.AdamW(self.model.parameters(), lr=kwargs['learning_rate'],
                                weight_decay=kwargs['weight_decay'])
        criterion = nn.MSELoss()
        val = []
        tr = []
        te = []
        minval = 100
        minvep = -1

        for epoch in range(kwargs['epoch']):
            if verbose > 1:
                logger.write(f'epoch:  {epoch}\n')

            self.model.train()
            running_loss = 0

            # for input_nodes, output_nodes, blocks in dataloader:

            self.model.train()
            for i, batch_idx in enumerate(dataloader):
                # feature_sampled = np.random.choice(g.nodes('feature'), 0.5*len(g.nodes('feature'), replace=False),
                #                  p=feature_weight)
                if self.args.node_sampling_rate < 1:
                    feature_sampled = torch.multinomial(feature_weight,
                                                        int(self.args.node_sampling_rate * len(g.nodes('feature'))),
                                                        replacement=False)
                else:
                    feature_sampled = torch.arange(len(g.nodes('feature')))
                subgraph = dgl.node_subgraph(g, {
                    'cell': batch_idx,
                    'feature': feature_sampled,
                }).to(self.args.device)  # XXX: bottlenect
                logits = self.model(subgraph)
                output_labels = subgraph.nodes['cell'].data['label'].float()

                # blocks = [b.to(torch.device(self.args.device)) for b in blocks]
                # logits = self.model(blocks, sampled = True)

                # output_labels = blocks[-1].dstdata['label']['cell']

                loss = criterion(logits, output_labels)

                running_loss += loss.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

                del subgraph
                del output_labels
                del loss
                torch.cuda.empty_cache()
            tr.append(math.sqrt(running_loss / len(dataloader)))

            if epoch % eval_interval == 0:
                val.append(self.score(g_origin, split['valid'], y[split['valid']], 'cpu'))

                if verbose > 1:
                    logger.write(f'training loss:  {tr[-1]}\n')
                    logger.flush()
                    logger.write(f'validation loss:  {val[-1]}\n')
                    logger.flush()

                if eval:
                    te.append(self.score(g_origin, np.arange(TRAIN_SIZE, CELL_SIZE), y_test, 'cpu'))
                    if verbose > 1:
                        logger.write(f'testing loss:  {te[-1]}\n')
                        logger.flush()

                if val[-1] < minval:
                    minval = val[-1]
                    minvep = epoch // eval_interval
                    if kwargs['save_best']:
                        torch.save(self.model, f'{kwargs["model_folder"]}/{PREFIX}.best.pth')
                    best_dict = deepcopy(self.model.state_dict())

                if epoch > 1500 and kwargs['early_stopping'] > 0 and min(val[-kwargs['early_stopping']:]) > minval:
                    if verbose > 1:
                        logger.write('Early stopped.\n')
                    break

                if epoch > 1200:
                    if epoch % 15 == 0:
                        for p in opt.param_groups:
                            p['lr'] *= kwargs['lr_decay']

                if verbose > 0:
                    print('epoch', epoch)
                    print('training: ', tr[-1])
                    print('valid: ', val[-1])
                    if eval:
                        print('testing: ', te[-1])

            torch.cuda.empty_cache()

        if kwargs['save_final']:
            state = {'model': self.model, 'optimizer': opt.state_dict(), 'epoch': epoch - 1}
            torch.save(state, f'{kwargs["model_folder"]}/{PREFIX}.epoch{epoch}.pth')

        if verbose > 1:
            if eval:
                logger.write(
                    f'epoch {minvep} minimal val {minval} with training:  {tr[minvep]} and testing: {te[minvep]}\n')
            else:
                logger.write(f'epoch {minvep} minimal val {minval} with training:  {tr[minvep]}\n')
            logger.close()

        if verbose > 0 and eval:
            print('min testing', min(te), te.index(min(te)))
            print('converged testing', minvep * eval_interval, te[minvep])
        self.model.load_state_dict(best_dict)
        return self.model


class ScMoGCN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.npw = not args.pathway
        self.nrc = args.no_readout_concatenate

        hid_feats = args.hidden_size
        out_feats = args.OUTPUT_SIZE
        FEATURE_SIZE = args.FEATURE_SIZE

        if not args.no_batch_features:
            self.extra_encoder = nn.Linear(args.BATCH_NUM, hid_feats)

        if args.cell_init == 'none':
            self.embed_cell = nn.Embedding(2, hid_feats)
        else:
            self.embed_cell = nn.Linear(100, hid_feats)

        self.embed_feat = nn.Embedding(FEATURE_SIZE, hid_feats)

        self.input_linears = nn.ModuleList()
        self.input_acts = nn.ModuleList()
        self.input_norm = nn.ModuleList()
        for i in range((args.embedding_layers - 1) * 2):
            self.input_linears.append(nn.Linear(hid_feats, hid_feats))
        if args.activation == 'gelu':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_acts.append(nn.LeakyReLU())
        if args.normalization == 'batch':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_norm.append(nn.BatchNorm1d(hid_feats))
        elif args.normalization == 'layer':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_norm.append(nn.LayerNorm(hid_feats))
        elif args.normalization == 'group':
            for i in range((args.embedding_layers - 1) * 2):
                self.input_norm.append(nn.GroupNorm(4, hid_feats))

        if self.npw:
            self.edges = ['feature2cell', 'cell2feature']
        else:
            self.edges = ['feature2cell', 'cell2feature', 'pathway']

        self.conv_layers = nn.ModuleList()
        if args.residual == 'res_cat':
            self.conv_layers.append(
                dglnn.HeteroGraphConv(
                    dict(
                        zip(self.edges, [
                            dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type=args.agg_function,
                                           norm=None) for i in range(len(self.edges))
                        ])), aggregate='stack'))
            for i in range(args.conv_layers - 1):
                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        dict(
                            zip(self.edges, [
                                dglnn.SAGEConv(in_feats=hid_feats * 2, out_feats=hid_feats,
                                               aggregator_type=args.agg_function, norm=None)
                                for i in range(len(self.edges))
                            ])), aggregate='stack'))

        else:
            for i in range(args.conv_layers):
                self.conv_layers.append(
                    dglnn.HeteroGraphConv(
                        dict(
                            zip(self.edges, [
                                dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats,
                                               aggregator_type=args.agg_function, norm=None)
                                for i in range(len(self.edges))
                            ])), aggregate='stack'))

        self.conv_acts = nn.ModuleList()
        self.conv_norm = nn.ModuleList()
        if args.activation == 'gelu':
            for i in range(args.conv_layers * 2):
                self.conv_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range(args.conv_layers * 2):
                self.conv_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range(args.conv_layers * 2):
                self.conv_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range(args.conv_layers * 2):
                self.conv_acts.append(nn.LeakyReLU())

        if args.normalization == 'batch':
            for i in range(args.conv_layers * len(self.edges)):
                self.conv_norm.append(nn.BatchNorm1d(hid_feats))
        elif args.normalization == 'layer':
            for i in range(args.conv_layers * len(self.edges)):
                self.conv_norm.append(nn.LayerNorm(hid_feats))
        elif args.normalization == 'group':
            for i in range(args.conv_layers * len(self.edges)):
                self.conv_norm.append(nn.GroupNorm(4, hid_feats))

        self.att_linears = nn.ModuleList()
        if args.pathway_aggregation == 'attention':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats, hid_feats))
        elif args.pathway_aggregation == 'one_gate':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats * 3, hid_feats))
        elif args.pathway_aggregation == 'two_gate':
            for i in range(args.conv_layers * 2):
                self.att_linears.append(nn.Linear(hid_feats * 2, hid_feats))
        elif args.pathway_aggregation == 'cat':
            for i in range(args.conv_layers):
                self.att_linears.append(nn.Linear(hid_feats * 2, hid_feats))

        self.readout_linears = nn.ModuleList()
        self.readout_acts = nn.ModuleList()

        if args.weighted_sum:
            print("Weighted_sum enabled. Argument '--no_readout_concatenate' won't take effect.")
            for i in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        elif self.nrc:
            for i in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats, hid_feats))
            self.readout_linears.append(nn.Linear(hid_feats, out_feats))
        else:
            for i in range(args.readout_layers - 1):
                self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, hid_feats * args.conv_layers))
            self.readout_linears.append(nn.Linear(hid_feats * args.conv_layers, out_feats))

        if args.activation == 'gelu':
            for i in range(args.readout_layers - 1):
                self.readout_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range(args.readout_layers - 1):
                self.readout_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range(args.readout_layers - 1):
                self.readout_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range(args.readout_layers - 1):
                self.readout_acts.append(nn.LeakyReLU())

        self.wt = nn.Parameter(torch.zeros(args.conv_layers))
        if args.pathway_aggregation == 'alpha' and args.pathway_alpha < 0:
            self.aph = nn.Parameter(torch.zeros(2))

    def attention_agg(self, layer, h0, h):
        # h: h^{l-1}, dimension: (batch, hidden)
        # feats: result from two conv(cell conv and pathway conv), stacked together; dimension: (batch, 2, hidden)
        args = self.args
        if h.shape[1] == 1:
            return self.conv_norm[layer * len(self.edges) + 1](h.squeeze(1))
        elif args.pathway_aggregation == 'sum':
            return h[:, 0, :] + h[:, 1, :]
        else:
            h1 = h[:, 0, :]
            h2 = h[:, 1, :]

            if args.subpath_activation:
                h1 = F.leaky_relu(h1)
                h2 = F.leaky_relu(h2)

            h1 = self.conv_norm[layer * len(self.edges) + 1](h1)
            h2 = self.conv_norm[layer * len(self.edges) + 2](h2)

        if args.pathway_aggregation == 'attention':
            feats = torch.stack([h1, h2], 1)
            att = torch.transpose(F.softmax(torch.matmul(feats, self.att_linears[layer](h0).unsqueeze(-1)), 1), 1, 2)
            feats = torch.matmul(att, feats)
            return feats.squeeze(1)
        elif args.pathway_aggregation == 'one_gate':
            att = torch.sigmoid(self.att_linears[layer](torch.cat([h0, h1, h2], 1)))
            return att * h1 + (1 - att) * h2
        elif args.pathway_aggregation == 'two_gate':
            att1 = torch.sigmoid(self.att_linears[layer * 2](torch.cat([h0, h1], 1)))
            att2 = torch.sigmoid(self.att_linears[layer * 2 + 1](torch.cat([h0, h2], 1)))
            return att1 * h1 + att2 * h2
        elif args.pathway_aggregation == 'alpha':
            if args.pathway_alpha < 0:
                weight = torch.softmax(self.aph, -1)
                return weight[0] * h1 + weight[1] * h2
            else:
                return (1 - args.pathway_alpha) * h1 + args.pathway_alpha * h2
        elif args.pathway_aggregation == 'cat':
            return self.att_linears[layer](torch.cat([h1, h2], 1))

    def conv(self, graph, layer, h, hist):
        args = self.args
        h0 = hist[-1]
        h = self.conv_layers[layer](graph, h, mod_kwargs=dict(
            zip(self.edges, [{
                'edge_weight':
                F.dropout(graph.edges[self.edges[i]].data['weight'], p=args.edge_dropout, training=self.training)
            } for i in range(len(self.edges))])))

        if args.model_dropout > 0:
            h = {
                'feature':
                F.dropout(self.conv_acts[layer * 2](self.attention_agg(layer, h0['feature'], h['feature'])),
                          p=args.model_dropout, training=self.training),
                'cell':
                F.dropout(self.conv_acts[layer * 2 + 1](self.conv_norm[layer * len(self.edges)](h['cell'].squeeze(1))),
                          p=args.model_dropout, training=self.training)
            }
        else:
            h = {
                'feature': self.conv_acts[layer * 2](self.attention_agg(layer, h0['feature'], h['feature'])),
                'cell': self.conv_acts[layer * 2 + 1](self.conv_norm[layer * len(self.edges)](h['cell'].squeeze(1)))
            }

        return h

    def calculate_initial_embedding(self, graph):
        args = self.args

        input1 = F.leaky_relu(self.embed_feat(graph.srcdata['id']['feature']))
        input2 = F.leaky_relu(self.embed_cell(graph.srcdata['id']['cell']))

        if not args.no_batch_features:
            batch_features = graph.srcdata['bf']['cell']
            input2 += F.leaky_relu(F.dropout(self.extra_encoder(batch_features), p=0.2,
                                             training=self.training))[:input2.shape[0]]

        hfeat = input1
        hcell = input2
        for i in range(args.embedding_layers - 1, (args.embedding_layers - 1) * 2):
            hfeat = self.input_linears[i](hfeat)
            hfeat = self.input_acts[i](hfeat)
            if args.normalization != 'none':
                hfeat = self.input_norm[i](hfeat)
            if args.model_dropout > 0:
                hfeat = F.dropout(hfeat, p=args.model_dropout, training=self.training)

        for i in range(args.embedding_layers - 1):
            hcell = self.input_linears[i](hcell)
            hcell = self.input_acts[i](hcell)
            if args.normalization != 'none':
                hcell = self.input_norm[i](hcell)
            if args.model_dropout > 0:
                hcell = F.dropout(hcell, p=args.model_dropout, training=self.training)

        return hfeat, hcell

    def propagate_with_sampling(self, blocks):
        args = self.args
        hfeat, hcell = self.calculate_initial_embedding(blocks[0])

        h = {'feature': hfeat, 'cell': hcell}

        for i in range(args.conv_layers):

            if i > 0:
                hfeat0, hcell0 = self.calculate_initial_embedding(blocks[i])

                h = {'feature': torch.cat([h['feature'], hfeat0], 1), 'cell': torch.cat([h['cell'], hcell0], 1)}

            hist = [h]
            h = self.conv(blocks[i], i, h, hist)

        hist = [h] * (args.conv_layers + 1)
        return hist  # , hist[-1]['feature']

    def propagate(self, graph):
        args = self.args
        hfeat, hcell = self.calculate_initial_embedding(graph)

        h = {'feature': hfeat, 'cell': hcell}
        hist = [h]

        for i in range(args.conv_layers):
            if i == 0 or args.residual == 'none':
                pass
            elif args.residual == 'res_add':
                if args.initial_residual:
                    h = {'feature': h['feature'] + hist[0]['feature'], 'cell': h['cell'] + hist[0]['cell']}

                else:
                    h = {'feature': h['feature'] + hist[-2]['feature'], 'cell': h['cell'] + hist[-2]['cell']}

            elif args.residual == 'res_cat':
                if args.initial_residual:
                    h = {
                        'feature': torch.cat([h['feature'], hist[0]['feature']], 1),
                        'cell': torch.cat([h['cell'], hist[0]['cell']], 1)
                    }
                else:
                    h = {
                        'feature': torch.cat([h['feature'], hist[-2]['feature']], 1),
                        'cell': torch.cat([h['cell'], hist[-2]['cell']], 1)
                    }

            h = self.conv(graph, i, h, hist)
            hist.append(h)

        return hist  #, hist[-1]['feature']

    def forward(self, graph, sampled=False):
        args = self.args
        if sampled:
            hist = self.propagate_with_sampling(graph)
        else:
            hist = self.propagate(graph)

        if args.weighted_sum:
            h = 0
            weight = torch.softmax(self.wt, -1)
            for i in range(args.conv_layers):
                h += weight[i] * hist[i + 1]['cell']
        elif not self.nrc:
            h = torch.cat([i['cell'] for i in hist[1:]], 1)
        else:
            h = hist[-1]['cell']

        for i in range(args.readout_layers - 1):
            h = self.readout_linears[i](h)
            h = F.dropout(self.readout_acts[i](h), p=args.model_dropout, training=self.training)
        h = self.readout_linears[-1](h)

        if args.output_relu == 'relu':
            return F.relu(h)
        elif args.output_relu == 'leaky_relu':
            return F.leaky_relu(h)

        return h
