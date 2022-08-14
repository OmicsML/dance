"""Reimplementation of STAGATE.

Extended from https://github.com/QIFEIDKN/STAGATE

Reference
----------
Dong, Kangning, and Shihua Zhang. "Deciphering spatial domains from spatially resolved transcriptomics with an adaptive
graph attention auto-encoder." Nature communications 13.1 (2022): 1-12.

"""

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import sklearn.neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import mixture
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from tqdm import tqdm


def transfer_pytorch_data(adata, adj):
    edgeList = adj
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                    x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)


def mclust_P(adata, num_cluster, used_obsm='STAGATE', modelNames='EEE'):

    from sklearn import mixture
    g = mixture.GaussianMixture(n_components=num_cluster, covariance_type='tied', warm_start=True, n_init=100,
                                max_iter=300, reg_covar=1.4663143602030552e-04, random_state=36282,
                                tol=0.00022187708009762592)
    res = g.fit_predict(adata.obsm[used_obsm])
    adata.obs['mclust'] = res
    return adata


'''
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
'''


class GATConv(MessagePassing):
    """Graph attention layer from Graph Attention Network."""
    _alpha = None

    def __init__(self, in_channels, out_channels, heads: int = 1, concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(self, x, edge_index, size=None, return_attention_weights=None, attention=True, tied_attention=None):

        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)
            # return x[0].view(-1, self.heads * self.out_channels)

        if tied_attention == None:
            # Next, we compute node-level attention coefficients, both for source
            # and target nodes (if present):
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        # alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.sigmoid(alpha)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)


class Stagate(torch.nn.Module):
    """Stagate class.

    Parameters
    ----------
    hidden_dims : int
        hidden dimensions

    """

    def __init__(self, hidden_dims):
        super().__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):
        """forward function for training.

        Parameters
        ----------
        features :
            node features.
        edge_index :
            adjacent matrix.

        Returns
        -------
        h2 :
            the second hidden layer.
        h4 :
            the forth hidden layer.

        """
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return h2, h4  # F.log_softmax(x, dim=-1)

    def fit(self, adata, graph, n_epochs=1, lr=0.001, key_added='STAGATE', gradient_clipping=5., pre_resolution=0.2,
            weight_decay=0.0001, verbose=True, random_seed=0, save_loss=False, save_reconstrction=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        """fit function for training.

        Parameters
        ----------
        adata :
            input data.
        graph :
            graph structure.
        n_epochs : int optional
            number of epochs.
        lr : float optional
            learning rate.
        key_added : str optional
            by default 'STAGATE'.
        gradient_clipping : float optional
            gradient clipping.
        pre_resolution : float optional
            pre resolution.
        weight_decay : float optional
            weight decay.
        verbose : bool optional
            verbose, by default to be True.
        random_seed : int optional
            random seed by default to be 0.
        save_loss : bool optional
            by default to be False.
        save_reconstrction : bool optional
            by default to be False.
        device : str optional
            to indicate gpu or cpu device.

        Returns
        -------
        None.

        """
        adata.X = sp.csr_matrix(adata.X)

        if 'highly_variable' in adata.var.columns:
            adata_Vars = adata[:, adata.var['highly_variable']]
        else:
            adata_Vars = adata

        if verbose:
            print('Size of Input: ', adata_Vars.shape)
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        data = transfer_pytorch_data(adata_Vars, graph)

        model = self.to(device)
        data = data.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # loss_list = []
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            loss = F.mse_loss(data.x, out)
            # loss_list.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

        print("eval ....")
        model.eval()
        z, out = model(data.x, data.edge_index)

        STAGATE_rep = z.to('cpu').detach().numpy()
        adata.obsm[key_added] = STAGATE_rep

        if save_loss:
            adata.uns['STAGATE_loss'] = loss
        if save_reconstrction:
            ReX = out.to('cpu').detach().numpy()
            ReX[ReX < 0] = 0
            adata.layers['STAGATE_ReX'] = ReX

        print("post process...")
        sc.pp.neighbors(adata, use_rep='STAGATE')
        sc.tl.umap(adata)
        #adata = mclust_R(adata, used_obsm='STAGATE', num_cluster=7)
        adata = mclust_P(adata, used_obsm='STAGATE', num_cluster=7)
        self.adata = adata

    def predict(self, ):
        """prediction function.

        Parameters
        ----------

        Returns
        -------
        self.y_pred :
            predicted label.

        """
        data_dropna = self.adata.obs.dropna()
        self.y_pred = data_dropna['mclust']
        self.target = data_dropna['ground_truth']
        return data_dropna['mclust']

    def score(self, y_true=None):
        """score function to get score of prediction.

        Parameters
        ----------
        y_true :
            ground truth label.

        Returns
        -------
        score : float
            metric eval score.

        """
        from sklearn.metrics.cluster import adjusted_rand_score
        score = adjusted_rand_score(self.target, self.y_pred)
        print("ARI {}".format(adjusted_rand_score(self.target, self.y_pred)))
        return score
