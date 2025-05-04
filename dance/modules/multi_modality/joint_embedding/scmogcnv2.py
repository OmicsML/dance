import math
import os

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dance.utils import SimpleIndexDataset
from dance.utils.metrics import *


def propagation_layer_combination(X, idx, wt, from_logits=True):
    if from_logits:
        wt = torch.softmax(wt, -1)

    x = 0
    for i in range(wt.shape[0]):
        x += wt[i] * X[i][idx]

    return x


class ScMoGCN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.opw = args.only_pathway
        self.npw = args.no_pathway
        self.nrc = args.no_readout_concatenate

        hid_feats = args.hidden_size
        out_feats = args.FEATURE_SIZE
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

        for i in range((args.embedding_layers - 1) * 2):
            self.input_acts.append(nn.GELU())

        for i in range((args.embedding_layers - 1) * 2):
            self.input_norm.append(nn.GroupNorm(4, hid_feats))

        if self.opw:
            self.edges = ['feature2cell', 'pathway']
        elif self.npw:
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
        for i in range(args.conv_layers * 2):
            self.conv_acts.append(nn.GELU())

        for i in range(args.conv_layers * len(self.edges)):
            self.conv_norm.append(nn.GroupNorm(4, hid_feats))

        self.att_linears = nn.ModuleList()
        for i in range(args.conv_layers):
            self.att_linears.append(nn.Linear(hid_feats * 2, hid_feats))

        self.cc_decoder = nn.Linear(47, 20)
        self.decoder = nn.ModuleList()
        self.decoder_acts = nn.ModuleList()
        # TODO: remove this before releaing
        hid_feats = 20 + (54 - 45) + 9
        if args.weighted_sum:
            print("Weighted_sum enabled. Argument '--no_readout_concatenate' won't take effect.")
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats, hid_feats))
            self.decoder.append(nn.Linear(hid_feats, out_feats))
        elif self.nrc:
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats, hid_feats))
            self.decoder.append(nn.Linear(hid_feats, out_feats))
        else:
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats * args.conv_layers, hid_feats * args.conv_layers))
            self.decoder.append(nn.Linear(hid_feats * args.conv_layers, out_feats))

        for i in range(args.readout_layers - 1):
            self.decoder_acts.append(nn.GELU())

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

        return hist  # , hist[-1]['feature']

    def encode(self, graph, sampled=False):
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

        return h  #(h-torch.mean(h, 1, False).reshape(-1,1))/torch.std(h, 1, False).reshape(-1,1) #F.normalize(h)

    def forward(self, graph, sampled=False):
        return self.encode(graph, sampled)

    def decode(self, h):

        for i in range(self.args.readout_layers - 1):
            h = self.decoder[i](h)
            h = F.dropout(self.decoder_acts[i](h), p=self.args.model_dropout, training=self.training)
        h = self.decoder[-1](h)

        return h

    def embed(self, graph, sampled=False):
        emb = self.encode(graph, sampled)
        return torch.cat(
            [(emb[:, :45].detach() + emb[:, 45:90] + self.cc_decoder(emb[:, -2:].detach())) / 3, emb[:, 90:98]], 1)


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

    def fit(self, g, y, train_labels=None, epochs=500):
        """Fit function for training.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.
        y : torch.Tensor
            Modality features, used as labels for reconstruction.
        train_labels : torch.Tensor
            Training supervision, by default to be None.
        epochs : int optional
            Maximum number of training epochs, by default to be 500.

        Returns
        -------
        None.

        """

        g = g.long()
        idx = np.random.permutation(train_labels[0].shape[0])
        train_idx = idx[:int(idx.shape[0] * 0.9)]
        val_idx = idx[int(idx.shape[0] * 0.9):]

        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=5000,  #self.args.batch_size,
            shuffle=True,
            num_workers=1,
        )

        mse = nn.MSELoss()

        if train_labels is not None:
            c_decoder = nn.Linear(20, train_labels[0].max() + 1).to(self.args.device)
            # b_decoder = nn.Linear(self.args.hidden_size, train_labels[1].max()+1).to(self.args.device)
            cc_decoder = nn.Linear(20, train_labels[3].shape[1]).to(self.args.device)
            # cc_decoder = nn.Linear(train_labels[3].shape[1], train_labels[0].max()+1).to(self.args.device)
            train_labels = [torch.from_numpy(i).to(self.args.device) for i in train_labels]
            ce = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                [
                    {
                        'params': self.model.parameters()
                    },
                    {
                        'params': c_decoder.parameters()
                    },
                    # {'params': b_decoder.parameters()},
                    {
                        'params': cc_decoder.parameters()
                    }
                ],
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        vals = []
        feature_weight = g.in_degrees(etype='cell2feature').float()

        for epoch in range(epochs):
            self.model.train()
            print('epoch', epoch)

            # loss = mse(self.model.decode(self.model.encode(g))[train_idx], y[train_idx])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss = 0
            for iter, batch_idx in enumerate(train_loader):
                if True:
                    feature_sampled = torch.multinomial(feature_weight, int(0.6 * len(g.nodes('feature'))),
                                                        replacement=False).to(self.args.device)
                    subgraph = dgl.node_subgraph(g, {
                        'cell': batch_idx.to(self.args.device),
                        'feature': feature_sampled
                    })
                    # subgraph = g
                    emb = self.model.encode(subgraph)
                    # output = self.model.decode(emb)#[batch_idx]
                else:
                    emb = self.model.encode(subgraph)
                    output = self.model.decode(emb)[batch_idx]

                    # loss = mse(output, y[batch_idx])

                # l1 = mse(output[:, :self.args.feat1], y[batch_idx, :self.args.feat1])
                # l2 = mse(output[:, -self.args.feat2:], y[batch_idx, -self.args.feat2:])
                # loss = l1 * 0.5 + l2 * 0.5

                if train_labels is not None:
                    # l3 = ce(c_decoder(emb), train_labels[0][batch_idx].long())
                    # l4 = ce(b_decoder(emb), train_labels[1][batch_idx].long())
                    # l5 = mse(cc_decoder(emb), train_labels[3][batch_idx].float())
                    # emb[:, 45:54] = F.one_hot(train_labels[1][batch_idx])

                    temp = torch.cat([
                        emb[:, :20],
                        emb[:, 45:-2],
                        F.one_hot(train_labels[1][batch_idx].long()),
                    ], 1)

                    output = self.model.decode(temp)
                    del temp

                    l1 = mse(output[:, :self.args.feat1], y[batch_idx, :self.args.feat1])
                    l2 = mse(output[:, -self.args.feat2:], y[batch_idx, -self.args.feat2:])
                    loss = l1 * 0.5 + l2 * 0.5

                    # l3 = ce(emb[:, :train_labels[0].max()+1], train_labels[0][batch_idx].long())
                    # if epoch>1:
                    # with torch.no_grad():
                    #     print((torch.argmax(emb[:, :train_labels[0].max()+1], 1) == train_labels[0][batch_idx]).float().mean().item())
                    # l4 = ce(emb[:, train_labels[0].max()+1:train_labels[0].max()+train_labels[1].max()+2], train_labels[1][batch_idx].long())
                    with torch.no_grad():
                        print((torch.argmax(c_decoder(emb[:, :20]),
                                            1) == train_labels[0][batch_idx]).float().mean().item())
                    l3 = ce(c_decoder(emb[:, :20]), train_labels[0][batch_idx].long())
                    l5 = mse(cc_decoder(emb[:, :20]), train_labels[3][batch_idx].float())
                    # l5 = mse(emb[:, -2:], train_labels[3][batch_idx].float())
                    # l6 = torch.linalg.norm(emb[:, 45: 90])
                    # # l5 = mse(cc_decoder(emb), train_labels[3][batch_idx].float())
                    # loss += l3 + l5
                    loss += l3 + l5
                    # del l3, l4, l5, l6
                    del l3, l5
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del subgraph, output, loss, emb, l1, l2
                torch.cuda.empty_cache()
            print(f'train loss: ', total_loss / len(train_loader))

            vals.append(self.score(g, y, train_labels, val_idx, c_decoder))
            print('val loss', vals[-1])

            if min(vals) == vals[-1]:
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(self.model.state_dict(), f'models/model_joint_embedding_{self.args.seed}.pth')

            if epoch > self.args.early_stopping and min(vals) != min(vals[-self.args.early_stopping:]):
                print('Early stopped.')
                break

            if epoch > 150:
                for p in optimizer.param_groups:
                    p['lr'] *= self.args.lr_decay

        return self.model

    def to(self, device):
        """Performs device conversion.

        Parameters
        ----------
        device : str
            Target device.

        Returns
        -------
        self : ScMoGCNWrapper
            Converted model.

        """
        self.args.device = device
        self.model = self.model.to(device)
        return self

    def load(self, path, map_location=None):
        """Load model parameters from checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        map_location : str optional
            Mapped device. This parameter will be passed to torch.load function if not none.

        Returns
        -------
        None.

        """

        if map_location is not None:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(path))

    def predict(self, g):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.

        Returns
        -------
        prediction : torch.Tensor
            Joint embedding of input data.

        """

        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode(g)
            # print(h[:, -2:])
            # print(h[0])
            # prediction = (h - torch.mean(h, 1, False).reshape(-1, 1)) / torch.std(h, 1, False).reshape(-1, 1)
            # prediction = torch.cat([F.softmax(h[:, :45], 1), h[:, -10:]], 1)
            prediction = torch.cat([emb[:, :20], emb[:, 45:-2]], 1)
            return prediction
            # return h

    def score(self, g, y, train_labels, idx, c_decoder):
        """Score function to get score of prediction.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.
        y : torch.Tensor
            Modality features, used as labels for reconstruction.
        idx : Iterable[int]
            Index of testing samples for scoring.

        Returns
        -------
        loss : float
            Reconstruction loss.

        """

        self.model.eval()
        mse = nn.MSELoss()

        with torch.no_grad():
            # output = self.model.decode(self.model.encode(g))
            emb = self.model.encode(g)[idx]
            temp = torch.cat([
                emb[:, :20],
                emb[:, 45:-2],
                F.one_hot(train_labels[1][idx].long()),
            ], 1)
            # print((torch.argmax(emb[:, :train_labels[0].max() + 1], 1) == train_labels[0][idx]).float().mean().item())
            print((torch.argmax(c_decoder(emb[:, :20]), 1) == train_labels[0][idx]).float().mean().item())

            output = self.model.decode(temp)
            del temp

            l1 = mse(output[:, :self.args.feat1], y[idx, :self.args.feat1])
            l2 = mse(output[:, -self.args.feat2:], y[idx, -self.args.feat2:])
            loss = l1 * 0.5 + l2 * 0.5
            loss = math.sqrt(loss.item())
            # loss = mse(self.model.decode(self.model.encode(g))[idx], y[idx])
            del emb, output, l1, l2
            return loss
        # return loss1, loss2, loss3, loss4


import math
import os

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dance.utils import SimpleIndexDataset
from dance.utils.metrics import *


def propagation_layer_combination(X, idx, wt, from_logits=True):
    if from_logits:
        wt = torch.softmax(wt, -1)

    x = 0
    for i in range(wt.shape[0]):
        x += wt[i] * X[i][idx]

    return x


class ScMoGCN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.opw = args.only_pathway
        self.npw = args.no_pathway
        self.nrc = args.no_readout_concatenate

        hid_feats = args.hidden_size
        out_feats = args.FEATURE_SIZE
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

        for i in range((args.embedding_layers - 1) * 2):
            self.input_acts.append(nn.GELU())

        for i in range((args.embedding_layers - 1) * 2):
            self.input_norm.append(nn.GroupNorm(4, hid_feats))

        if self.opw:
            self.edges = ['feature2cell', 'pathway']
        elif self.npw:
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
        for i in range(args.conv_layers * 2):
            self.conv_acts.append(nn.GELU())

        for i in range(args.conv_layers * len(self.edges)):
            self.conv_norm.append(nn.GroupNorm(4, hid_feats))

        self.att_linears = nn.ModuleList()
        for i in range(args.conv_layers):
            self.att_linears.append(nn.Linear(hid_feats * 2, hid_feats))

        self.cc_decoder = nn.Linear(47, 20)
        self.decoder = nn.ModuleList()
        self.decoder_acts = nn.ModuleList()
        # TODO: remove this before releaing
        hid_feats = 20 + (54 - 45) + 9
        if args.weighted_sum:
            print("Weighted_sum enabled. Argument '--no_readout_concatenate' won't take effect.")
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats, hid_feats))
            self.decoder.append(nn.Linear(hid_feats, out_feats))
        elif self.nrc:
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats, hid_feats))
            self.decoder.append(nn.Linear(hid_feats, out_feats))
        else:
            for i in range(args.readout_layers - 1):
                self.decoder.append(nn.Linear(hid_feats * args.conv_layers, hid_feats * args.conv_layers))
            self.decoder.append(nn.Linear(hid_feats * args.conv_layers, out_feats))

        for i in range(args.readout_layers - 1):
            self.decoder_acts.append(nn.GELU())

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

        return hist  # , hist[-1]['feature']

    def encode(self, graph, sampled=False):
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

        return h  #(h-torch.mean(h, 1, False).reshape(-1,1))/torch.std(h, 1, False).reshape(-1,1) #F.normalize(h)

    def forward(self, graph, sampled=False):
        return self.encode(graph, sampled)

    def decode(self, h):

        for i in range(self.args.readout_layers - 1):
            h = self.decoder[i](h)
            h = F.dropout(self.decoder_acts[i](h), p=self.args.model_dropout, training=self.training)
        h = self.decoder[-1](h)

        return h

    def embed(self, graph, sampled=False):
        emb = self.encode(graph, sampled)
        return torch.cat(
            [(emb[:, :45].detach() + emb[:, 45:90] + self.cc_decoder(emb[:, -2:].detach())) / 3, emb[:, 90:98]], 1)


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

    def fit(self, g, y, train_labels=None, epochs=500):
        """Fit function for training.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.
        y : torch.Tensor
            Modality features, used as labels for reconstruction.
        train_labels : torch.Tensor
            Training supervision, by default to be None.
        epochs : int optional
            Maximum number of training epochs, by default to be 500.

        Returns
        -------
        None.

        """

        g = g.long()
        idx = np.random.permutation(train_labels[0].shape[0])
        train_idx = idx[:int(idx.shape[0] * 0.9)]
        val_idx = idx[int(idx.shape[0] * 0.9):]

        train_dataset = SimpleIndexDataset(train_idx)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=5000,  #self.args.batch_size,
            shuffle=True,
            num_workers=1,
        )

        mse = nn.MSELoss()

        if train_labels is not None:
            c_decoder = nn.Linear(20, train_labels[0].max() + 1).to(self.args.device)
            # b_decoder = nn.Linear(self.args.hidden_size, train_labels[1].max()+1).to(self.args.device)
            cc_decoder = nn.Linear(20, train_labels[3].shape[1]).to(self.args.device)
            # cc_decoder = nn.Linear(train_labels[3].shape[1], train_labels[0].max()+1).to(self.args.device)
            train_labels = [torch.from_numpy(i).to(self.args.device) for i in train_labels]
            ce = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                [
                    {
                        'params': self.model.parameters()
                    },
                    {
                        'params': c_decoder.parameters()
                    },
                    # {'params': b_decoder.parameters()},
                    {
                        'params': cc_decoder.parameters()
                    }
                ],
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        vals = []
        feature_weight = g.in_degrees(etype='cell2feature').float()

        for epoch in range(epochs):
            self.model.train()
            print('epoch', epoch)

            # loss = mse(self.model.decode(self.model.encode(g))[train_idx], y[train_idx])
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            total_loss = 0
            for iter, batch_idx in enumerate(train_loader):
                if True:
                    feature_sampled = torch.multinomial(feature_weight, int(0.6 * len(g.nodes('feature'))),
                                                        replacement=False).to(self.args.device)
                    subgraph = dgl.node_subgraph(g, {
                        'cell': batch_idx.to(self.args.device),
                        'feature': feature_sampled
                    })
                    # subgraph = g
                    emb = self.model.encode(subgraph)
                    # output = self.model.decode(emb)#[batch_idx]
                else:
                    emb = self.model.encode(subgraph)
                    output = self.model.decode(emb)[batch_idx]

                    # loss = mse(output, y[batch_idx])

                # l1 = mse(output[:, :self.args.feat1], y[batch_idx, :self.args.feat1])
                # l2 = mse(output[:, -self.args.feat2:], y[batch_idx, -self.args.feat2:])
                # loss = l1 * 0.5 + l2 * 0.5

                if train_labels is not None:
                    # l3 = ce(c_decoder(emb), train_labels[0][batch_idx].long())
                    # l4 = ce(b_decoder(emb), train_labels[1][batch_idx].long())
                    # l5 = mse(cc_decoder(emb), train_labels[3][batch_idx].float())
                    # emb[:, 45:54] = F.one_hot(train_labels[1][batch_idx])

                    temp = torch.cat([
                        emb[:, :20],
                        emb[:, 45:-2],
                        F.one_hot(train_labels[1][batch_idx].long()),
                    ], 1)

                    output = self.model.decode(temp)
                    del temp

                    l1 = mse(output[:, :self.args.feat1], y[batch_idx, :self.args.feat1])
                    l2 = mse(output[:, -self.args.feat2:], y[batch_idx, -self.args.feat2:])
                    loss = l1 * 0.5 + l2 * 0.5

                    # l3 = ce(emb[:, :train_labels[0].max()+1], train_labels[0][batch_idx].long())
                    # if epoch>1:
                    # with torch.no_grad():
                    #     print((torch.argmax(emb[:, :train_labels[0].max()+1], 1) == train_labels[0][batch_idx]).float().mean().item())
                    # l4 = ce(emb[:, train_labels[0].max()+1:train_labels[0].max()+train_labels[1].max()+2], train_labels[1][batch_idx].long())
                    with torch.no_grad():
                        print((torch.argmax(c_decoder(emb[:, :20]),
                                            1) == train_labels[0][batch_idx]).float().mean().item())
                    l3 = ce(c_decoder(emb[:, :20]), train_labels[0][batch_idx].long())
                    l5 = mse(cc_decoder(emb[:, :20]), train_labels[3][batch_idx].float())
                    # l5 = mse(emb[:, -2:], train_labels[3][batch_idx].float())
                    # l6 = torch.linalg.norm(emb[:, 45: 90])
                    # # l5 = mse(cc_decoder(emb), train_labels[3][batch_idx].float())
                    # loss += l3 + l5
                    loss += l3 + l5
                    # del l3, l4, l5, l6
                    del l3, l5
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                del subgraph, output, loss, emb, l1, l2
                torch.cuda.empty_cache()
            print(f'train loss: ', total_loss / len(train_loader))

            vals.append(self.score(g, y, train_labels, val_idx, c_decoder))
            print('val loss', vals[-1])

            if min(vals) == vals[-1]:
                if not os.path.exists('models'):
                    os.mkdir('models')
                torch.save(self.model.state_dict(), f'models/model_joint_embedding_{self.args.seed}.pth')

            if epoch > self.args.early_stopping and min(vals) != min(vals[-self.args.early_stopping:]):
                print('Early stopped.')
                break

            if epoch > 150:
                for p in optimizer.param_groups:
                    p['lr'] *= self.args.lr_decay

        return self.model

    def to(self, device):
        """Performs device conversion.

        Parameters
        ----------
        device : str
            Target device.

        Returns
        -------
        self : ScMoGCNWrapper
            Converted model.

        """
        self.args.device = device
        self.model = self.model.to(device)
        return self

    def load(self, path, map_location=None):
        """Load model parameters from checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        map_location : str optional
            Mapped device. This parameter will be passed to torch.load function if not none.

        Returns
        -------
        None.

        """

        if map_location is not None:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(path))

    def predict(self, g):
        """Predict function to get latent representation of data.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.

        Returns
        -------
        prediction : torch.Tensor
            Joint embedding of input data.

        """

        self.model.eval()
        with torch.no_grad():
            emb = self.model.encode(g)
            # print(h[:, -2:])
            # print(h[0])
            # prediction = (h - torch.mean(h, 1, False).reshape(-1, 1)) / torch.std(h, 1, False).reshape(-1, 1)
            # prediction = torch.cat([F.softmax(h[:, :45], 1), h[:, -10:]], 1)
            prediction = torch.cat([emb[:, :20], emb[:, 45:-2]], 1)
            return prediction
            # return h

    def score(self, g, y, train_labels, idx, c_decoder):
        """Score function to get score of prediction.

        Parameters
        ----------
        g : dgl.DGLGraph
            Constructed cell-feature graph.
        y : torch.Tensor
            Modality features, used as labels for reconstruction.
        idx : Iterable[int]
            Index of testing samples for scoring.

        Returns
        -------
        loss : float
            Reconstruction loss.

        """

        self.model.eval()
        mse = nn.MSELoss()

        with torch.no_grad():
            # output = self.model.decode(self.model.encode(g))
            emb = self.model.encode(g)[idx]
            temp = torch.cat([
                emb[:, :20],
                emb[:, 45:-2],
                F.one_hot(train_labels[1][idx].long()),
            ], 1)
            # print((torch.argmax(emb[:, :train_labels[0].max() + 1], 1) == train_labels[0][idx]).float().mean().item())
            print((torch.argmax(c_decoder(emb[:, :20]), 1) == train_labels[0][idx]).float().mean().item())

            output = self.model.decode(temp)
            del temp

            l1 = mse(output[:, :self.args.feat1], y[idx, :self.args.feat1])
            l2 = mse(output[:, -self.args.feat2:], y[idx, -self.args.feat2:])
            loss = l1 * 0.5 + l2 * 0.5
            loss = math.sqrt(loss.item())
            # loss = mse(self.model.decode(self.model.encode(g))[idx], y[idx])
            del emb, output, l1, l2
            return loss
        # return loss1, loss2, loss3, loss4
