# Standard library imports
import csv
import math
import os
import pickle as pkl
import random
import sys
from collections import Counter
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import torch
import torch as t
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils import data as tdata
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, _LRScheduler

# Local/third-party specific imports
import warnings
from dance.modules.base import BaseClassificationMethod
from dance.transforms import BaseTransform, Compose, NormalizeTotalLog1P, SetConfig
from dance.transforms.graph import StringDBGraph
from dance.typing import LogLevel, Tuple

warnings.filterwarnings("ignore")

# ==========================================
# 1. Data Handling (Dataset & DataLoader)
# ==========================================

def collate_func(batch):
    data0 = batch[0]
    if isinstance(data0, Data):
        tmp_x = [xx['x'] for xx in batch]
        tmp_y = [xx['y'] for xx in batch]
    elif isinstance(data0, (list, tuple)):
        tmp_x = [xx[0] for xx in batch]
        tmp_y = [xx[1] for xx in batch]

    tmp_data = Data()
    tmp_data['x'] = t.stack(tmp_x, dim=1)
    tmp_data['y'] = t.cat(tmp_y)
    tmp_data['edge_index'] = data0.edge_index
    tmp_data['batch'] = t.zeros_like(tmp_data['y'])
    tmp_data['num_graphs'] = 1
    return tmp_data


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], **kwargs):
        if 'collate_fn' not in kwargs.keys():
            kwargs['collate_fn'] = collate_func
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, **kwargs)


class ExprDataset(tdata.Dataset):
    def __init__(self, Expr, edge, y, device='cpu'): 
        super(ExprDataset, self).__init__()

        print('processing dataset (on CPU)...')
        self.gene_num = Expr.shape[1]
        
        # ==================== 修复部分开始 ====================
        # 处理 Edge (保持在 CPU)
        if isinstance(edge, list):
            print('multi graphs:', len(edge))
            # 这里的处理逻辑较复杂，假设多图也是由 Tensor 组成的列表
            processed_edges = []
            self.edge_num = []
            for x in edge:
                e_tensor = t.tensor(x).long().cpu() if not isinstance(x, t.Tensor) else x.long().cpu()
                # 检查并修正形状：确保是 [2, N]
                if e_tensor.shape[0] != 2 and e_tensor.shape[1] == 2:
                    e_tensor = e_tensor.t()
                processed_edges.append(e_tensor)
                self.edge_num.append(e_tensor.shape[1])
            self.common_edge = processed_edges

        elif isinstance(edge, (np.ndarray, t.Tensor)):
            # 转换为 Tensor
            edge_tensor = t.tensor(edge).long().cpu() if not isinstance(edge, t.Tensor) else edge.long().cpu()
            
            # 【关键修复】: 如果形状是 [N, 2] (比如 [899420, 2])，强制转置为 [2, N]
            if edge_tensor.shape[0] != 2 and edge_tensor.shape[1] == 2:
                print(f"Warning: Transposing edge_index from {edge_tensor.shape} to ({edge_tensor.shape[1]}, {edge_tensor.shape[0]}) for PyG compat.")
                edge_tensor = edge_tensor.t()
            
            self.common_edge = edge_tensor
            # 现在可以安全地获取边数 (第1维)
            self.edge_num = self.common_edge.shape[1]
        # ==================== 修复部分结束 ====================

        # 1. 处理 Expr
        if isinstance(Expr, np.ndarray):
            self.Expr = t.from_numpy(Expr).float()
        elif isinstance(Expr, t.Tensor):
            self.Expr = Expr.float().cpu()
        else:
            self.Expr = t.tensor(Expr).float()
            
        # 2. 处理 Label
        self.y = t.tensor(y).long().cpu()

        self.num_sam = len(self.y)
        self.sample_mapping_list = np.arange(self.num_sam)

        if len(self.Expr.shape) == 2:
            self.num_expr_feaure = 1
        else:
            self.num_expr_feaure = self.Expr.shape[2]

    def duplicate_minor_types(self, dup_odds=50, random_seed=2240):
        # 使用 numpy 操作 CPU 数据
        y_np = self.y.numpy()
        counter = Counter(y_np)
        max_num_types = max(counter.values())
        impute_indexs = np.arange(self.num_sam).tolist()

        np.random.seed(random_seed)
        
        for lab in np.unique(y_np):
            current_count = np.sum(y_np == lab)
            
            if max_num_types / current_count > dup_odds:
                impute_size = int(max_num_types / dup_odds) - current_count
                print('duplicate #celltype %d with %d cells' % (lab, impute_size))
                
                impute_idx = np.random.choice(np.where(y_np == lab)[0], size=impute_size, replace=True).tolist()
                impute_indexs += impute_idx

        impute_indexs = np.random.permutation(impute_indexs)
        print('org/imputed #cells:', self.num_sam, len(impute_indexs))
        self.num_sam = len(impute_indexs)
        self.sample_mapping_list = impute_indexs

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = self.sample_mapping_list[idx]
            data = self.get(idx)
            return data
        raise IndexError('Only integers are valid indices (got {}).'.format(type(idx).__name__))

    def split(self, idx):
        # 确保 idx 是 tensor
        if not isinstance(idx, t.Tensor):
            idx = t.tensor(idx).long()
        return ExprDataset(self.Expr[idx, :], self.common_edge, self.y[idx])
    
    def __len__(self):
        return self.num_sam

    def get(self, index):
        # CPU 上的快速操作
        data = Data()
        data['x'] = self.Expr[index, :].reshape([-1, self.num_expr_feaure])
        data['y'] = self.y[index].reshape([1, 1])
        data['edge_index'] = self.common_edge
        return data

# ==========================================
# 2. Model Definition (Layers & GNN)
# ==========================================

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True, activate=False, alphas=[0.65, 0.35], shared_weight=False, aggr='mean', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)
        self.shared_weight = shared_weight
        self.activate = activate
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        if self.shared_weight:
            self.self_weight = self.weight
        else:
            self.self_weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.alphas = alphas

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        uniform(self.in_channels, self.bias)
        uniform(self.in_channels, self.self_weight)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        out = torch.matmul(x, self.self_weight)
        out2 = self.propagate(edge_index, size=size, x=x, edge_weight=edge_weight)
        return self.alphas[0] * out + self.alphas[1] * out2

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.activate:
            aggr_out = F.relu(aggr_out)

        if torch.is_tensor(aggr_out):
            aggr_out = torch.matmul(aggr_out, self.weight)
        else:
            aggr_out = (None if aggr_out[0] is None else torch.matmul(aggr_out[0], self.weight),
                        None if aggr_out[1] is None else torch.matmul(aggr_out[1], self.weight))
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def help_bn(bn1, x):
    x = x.permute(1, 0, 2)  # #samples x #nodes x #features
    x = bn1(x)
    x = x.permute(1, 0, 2)  # #nodes x #samples x #features
    return x

def sup_constrive(representations, label, T, device):
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    
    # Create mask
    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t())) - torch.eye(n, n).to(device)
    mask_no_sim = torch.ones_like(mask) - mask
    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)
    
    similarity_matrix = torch.exp(similarity_matrix / T)
    similarity_matrix = similarity_matrix * mask_dui_jiao_0.to(device)
    sim = mask * similarity_matrix
    no_sim = similarity_matrix - sim
    no_sim_sum = torch.sum(no_sim, dim=1)

    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)
    loss = mask_no_sim + loss + torch.eye(n, n).to(device)
    loss = -torch.log(loss + 1e-8)  # Add epsilon
    if len(torch.nonzero(loss)) > 0:
        loss = torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))
    else:
        loss = torch.tensor(0.0).to(device)
    return loss

class WeightFreezing(nn.Module):
    def __init__(self, input_dim, output_dim, shared_ratio=0.3, multiple=0):
        super(WeightFreezing, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        mask = torch.rand(input_dim, output_dim) < shared_ratio
        self.register_buffer('shared_mask', mask)
        self.register_buffer('independent_mask', ~mask)
        self.multiple = multiple

    def forward(self, x, shared_weight):
        combined_weight = torch.where(self.shared_mask, shared_weight*self.multiple, self.weight.t())
        output = F.linear(x, combined_weight.t(), self.bias)
        return output

class scRGCL(nn.Module):
    def __init__(self, in_channel=1, mid_channel=8, out_channel=2, num_nodes=2207, edge_num=151215, **args):
        super(scRGCL, self).__init__()
        self.mid_channel = mid_channel
        self.dropout_ratio = args.get('dropout_ratio', 0.3)
        print('model dropout ratio:', self.dropout_ratio)
        n_out_nodes = num_nodes
        self.global_conv1_dim = 4 * 3
        self.global_conv2_dim = args.get('global_conv2_dim', 4)
        input_dim = num_nodes
        self.c_layer = nn.Linear(input_dim, 63, bias=False)

        self.res_conv1 = t.nn.Conv2d(8, 24, [1, 1])
        self.res_bn1 = t.nn.BatchNorm2d(24)
        self.res_act1 = nn.Tanh()

        self.res_conv2 = t.nn.Conv2d(24, 48, [1, 1])
        self.res_bn2 = t.nn.BatchNorm2d(48)
        self.res_act2 = nn.Tanh()

        self.res_conv3 = t.nn.Conv2d(48, 24, [1, 1])
        self.res_bn3 = t.nn.BatchNorm2d(24)
        self.res_act3 = nn.Tanh()

        self.res_conv4 = t.nn.Conv2d(24, 8, [1, 1])
        self.res_bn4 = t.nn.BatchNorm2d(8)
        self.res_act4 = nn.Tanh()

        self.res_fc = nn.Linear(8, 8)

        self.conv1 = SAGEConv(in_channel, 8)
        self.conv2 = SAGEConv(8, mid_channel)

        self.bn1 = torch.nn.LayerNorm((num_nodes, 8))
        self.bn2 = torch.nn.LayerNorm((num_nodes, mid_channel))
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.global_conv1 = t.nn.Conv2d(mid_channel * 1, self.global_conv1_dim, [1, 1])
        self.global_bn1 = torch.nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()

        self.global_conv2 = t.nn.Conv2d(self.global_conv1_dim, self.global_conv2_dim, [1, 1])
        self.global_bn2 = torch.nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()

        last_feature_node = 64
        channel_list = [self.global_conv2_dim * n_out_nodes, 256, 64]
        if args.get('channel_list', False):
            channel_list = [self.global_conv2_dim * n_out_nodes, 128]
            last_feature_node = 128

        self.nn = []
        for idx, num in enumerate(channel_list[:-1]):
            self.nn.append(nn.Linear(channel_list[idx], channel_list[idx+1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx+1]))
            if self.dropout_ratio > 0:
                self.nn.append(nn.Dropout(0.3))
            self.nn.append(nn.ReLU())
        self.global_fc_nn = nn.Sequential(*self.nn)
        self.fc1 = nn.Linear(last_feature_node, out_channel)
        self.classifier = WeightFreezing(last_feature_node, out_channel, shared_ratio=0.35)

        self.shared_weights = nn.Parameter(torch.Tensor(out_channel, last_feature_node), requires_grad=False)
        self.bias = nn.Parameter(torch.Tensor(last_feature_node))

        nn.init.kaiming_uniform_(self.shared_weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.shared_weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        self.fixed_weight = self.shared_weights.t() * self.classifier.shared_mask

        self.edge_num = edge_num
        self.weight_edge_flag = True  
        if self.weight_edge_flag:
            self.edge_weight = nn.Parameter(t.ones(edge_num).float()*0.01)
        else:
            self.edge_weight = None
    
        self.reset_parameters()

    @property
    def device(self):
        return self.c_layer.weight.device
    
    def reset_parameters(self):
        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)
        nn.init.kaiming_normal_(self.res_conv1.weight, mode='fan_out')
        uniform(8, self.res_conv1.bias)
        nn.init.kaiming_normal_(self.res_conv2.weight, mode='fan_out')
        uniform(24, self.res_conv2.bias)
        nn.init.kaiming_normal_(self.res_conv3.weight, mode='fan_out')
        uniform(48, self.res_conv3.bias)
        nn.init.kaiming_normal_(self.res_conv4.weight, mode='fan_out')
        uniform(24, self.res_conv4.bias)
        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        uniform(self.mid_channel, self.global_conv1.bias)
        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        uniform(self.global_conv1_dim, self.global_conv2.bias)
        self.global_fc_nn.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self,data,get_latent_varaible=False,ui=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if self.weight_edge_flag:
            one_graph_edge_weight = torch.sigmoid(self.edge_weight)
            edge_weight = one_graph_edge_weight
        else:
            edge_weight = None 
            
        # 维度交换
        x = x.permute(1, 0, 2) 
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.act1(x)
        x = x.permute(1, 0, 2)

        x = help_bn(self.bn1, x)

        if ui == True:
            c_x = x.permute(1, 0, 2)
            avg_pool = F.avg_pool1d(c_x,kernel_size=8)
            max_pool = avg_pool.squeeze(dim=-1)
            c_layer = self.c_layer(max_pool)
            
            contrast_label = np.squeeze(data.y.cpu().numpy(), -1)
            contrast_label = torch.tensor(contrast_label).to(self.device)
            loss_hard = sup_constrive(c_layer, contrast_label, 0.07, self.device)
        else:
            loss_hard = 0

        res = x
        if self.dropout_ratio > 0: x = F.dropout(x, p=0.1, training=self.training)

        x = x.permute(1, 2, 0)
        x = x.unsqueeze(dim=-1)

        h1 = self.res_act1(self.res_conv1(x))
        h1 = self.res_bn1(h1)
        h1 = F.dropout(h1, p=0.3, training=self.training)

        h2 = self.res_act2(self.res_conv2(h1))
        h2 = self.res_bn2(h2)
        h2 = F.dropout(h2, p=0.3, training=self.training)

        h3 = self.res_act3(self.res_conv3(h2))
        h3 = self.res_bn3(h3)
        h3 = F.dropout(h3, p=0.3, training=self.training)

        h4 = h3 + h1
        x = self.res_act4(self.res_conv4(h4))
        x = self.res_bn4(x)
        x = x.squeeze(dim=-1)
        x = x.permute(2, 0, 1)
        x = x + res
        x = self.res_fc(x)

        x = x.permute(1, 0, 2)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.act2(x)
        x = x.permute(1, 0, 2)
        
        x = help_bn(self.bn2, x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = x.permute(1,2,0)
        x = x.unsqueeze(dim=-1)
        x = self.global_conv1(x)
        x = self.global_act1(x)
        x = self.global_bn1(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = self.global_conv2(x)
        x = self.global_act1(x)
        x = self.global_bn2(x)
        if self.dropout_ratio >0: x = F.dropout(x, p=0.3, training=self.training)
        x = x.squeeze(dim=-1)
        num_samples = x.shape[0]

        x = x .view(num_samples, -1)
        x = self.global_fc_nn(x)
        if get_latent_varaible:
            return x
        else:
            x = self.classifier(x, self.fixed_weight.to(x.device))
            return F.softmax(x, dim=-1), loss_hard

# ==========================================
# 3. Training & Utility Functions
# ==========================================

def edge_transform_func(org_edge):
    edge = org_edge
    edge = t.tensor(edge.T)
    edge = remove_self_loops(edge)[0]
    edge = edge.numpy()
    return edge

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2
    return loss

def train2(model, optimizer, train_loader, epoch, device, loss_fn=None, scheduler=None, verbose=False):
    model.train()
    loss_all = 0
    iters = len(train_loader)
    
    # 【修复】防止 Loader 为空导致的除零错误
    if iters == 0:
        print(f"Warning: Epoch {epoch} skipped because DataLoader is empty (Batch size > Dataset size with drop_last=True).")
        return 0

    for idx, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)

        if verbose:
            print(data.y.shape, data.edge_index.shape)
            
        optimizer.zero_grad()
        label = data.y.reshape(-1)
        
        all_output = model(data, ui=True)
        all_output1 = model(data, ui=True)
        
        output = all_output[0]
        output1 = all_output1[0]
        c_loss = all_output[1]
        
        if loss_fn is None:
            ce_loss = 0.5 * (F.cross_entropy(output1, label) + F.cross_entropy(output, label))
            kl_loss = compute_kl_loss(output1, output)
            loss = ce_loss + 0.75 * kl_loss
        else:
            ce_loss = (loss_fn(output, label) + loss_fn(output1, label)) * 0.5
            kl_loss = compute_kl_loss(output1, output)
            loss = ce_loss + 0.75 * kl_loss

        if model.edge_weight is not None:
            l2_loss = 0 
            if isinstance(model.edge_weight, nn.Module):
                for edge_weight in model.edge_weight:
                    l2_loss += 0.1 * t.mean((edge_weight)**2)
            elif isinstance(model.edge_weight, t.Tensor):
                l2_loss = 0.1 * t.mean((model.edge_weight)**2)
            loss += l2_loss

        loss = loss + c_loss.item() * 0.1
        loss.backward()
        
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        if not (scheduler is None):
            scheduler.step((epoch - 1) + idx / iters)

    return (loss_all / iters)

@torch.no_grad()
def test2(model, loader, predicts=False, device=None):
    model.eval()
    y_pred_list = []
    y_true_list = []
    y_output_list = [] 
    cell_list = []

    for data in loader:
        # 【关键修改】在测试循环内部移动数据
        data = data.to(device)

        all_output = model(data, ui=False)
        output = all_output[0]
        pred = output.argmax(dim=1)

        y_pred_list.append(pred)
        y_true_list.append(data.y)

        if predicts:
            y_output_list.append(output)
            cell_list.append(torch.squeeze(data.x))

    y_pred = torch.cat(y_pred_list).cpu().numpy()
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_true = y_true.flatten()

    acc = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    if predicts:
        y_output = torch.cat(y_output_list).cpu().numpy()
        cell_processed_list = []
        for c in cell_list:
            cell_processed_list.append(c.cpu().numpy().T)
        cell_all = np.vstack(cell_processed_list) if cell_processed_list else np.array([])
        return acc, f1, y_true, y_pred, y_output, cell_all
    else:
        return acc, f1

# ==========================================
# 4. Main Wrapper Class
# ==========================================

class scRGCLWrapper(BaseClassificationMethod):
    """
    Wrapper class for scRGCL model.
    """
    def __init__(self, 
                 dropout_ratio: float = 0.1,
                 weight_decay: float = 1e-4,
                 init_lr: float = 0.001,
                 min_lr: float = 1e-6,
                 max_epoch_stage1: int = 14,
                 max_epoch_stage2: int = 50,
                 seed: int = 42,
                 out_dir: str = './output',
                 device: torch.device = torch.device("cuda:0")):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.weight_decay = weight_decay
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.max_epoch_stage1 = max_epoch_stage1
        self.max_epoch_stage2 = max_epoch_stage2
        self.seed = seed
        self.out_dir = out_dir
        self.device = device
        self.model = None
        self.dataset_attr = {} 
        self.str_labels = []

        os.makedirs(os.path.join(self.out_dir, 'models'), exist_ok=True)

    @staticmethod
    def preprocessing_pipeline(log_level="INFO", thres=0.99, species="human"):
        return Compose(
            NormalizeTotalLog1P(),
            StringDBGraph(thres=thres, species=species),
            SetConfig({
                "label_channel": "cell_type"
            }),
            log_level=log_level,
        )


    def fit(self, adata, batch_size=64):
        logExpr = adata.X
        if scipy.sparse.issparse(logExpr):
            print("Converting sparse matrix to dense array for speed...")
            logExpr = logExpr.toarray()
        
        logExpr = logExpr.astype(np.float32)
        
        if 'str_labels' in adata.uns:
            self.str_labels = adata.uns['str_labels']
            label = np.array([self.str_labels.index(ct) for ct in adata.obs['cell_type']])
        else:
            unique_labels = np.unique(adata.obs['cell_type'])
            label_map = {k: v for v, k in enumerate(unique_labels)}
            self.str_labels = list(unique_labels)
            label = np.array([label_map[ct] for ct in adata.obs['cell_type']])

        if 'edge_index' not in adata.uns:
            raise ValueError("adata.uns['edge_index'] is missing.")
        
        used_edge = edge_transform_func(adata.uns['edge_index'])
        print('Data preparation complete.')

        label_type = np.unique(label.reshape(-1))
        alpha = np.array([np.sum(label == x) for x in label_type])
        alpha = np.max(alpha) / alpha
        alpha = np.clip(alpha, 1, 50)
        alpha = alpha / np.sum(alpha)
        
        loss_fn = t.nn.CrossEntropyLoss(weight=t.tensor(alpha).float())
        loss_fn = loss_fn.to(self.device)

        full_dataset = ExprDataset(Expr=logExpr, edge=used_edge, y=label, device='cpu')
        
        self.dataset_attr = {
            'gene_num': full_dataset.gene_num,
            'class_num': len(np.unique(label)),
            'num_expr_feature': full_dataset.num_expr_feaure,
            'edge_num': full_dataset.edge_num,
            'used_edge': used_edge
        }

        self.model = scRGCL(in_channel=self.dataset_attr['num_expr_feature'], 
                            num_nodes=self.dataset_attr['gene_num'],
                            out_channel=self.dataset_attr['class_num'], 
                            edge_num=self.dataset_attr['edge_num'],
                            dropout_ratio=self.dropout_ratio).to(self.device)
        
        print(f"Model initialized on {next(self.model.parameters()).device}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        # Stage 1 Training
        train_dataset = full_dataset
        train_dataset.duplicate_minor_types(dup_odds=50)
        
        # 【修复】如果 Stage 1 数据集太小，关闭 drop_last
        s1_drop_last = True
        if len(train_dataset) < batch_size:
            print(f"Notice: Stage 1 dataset size ({len(train_dataset)}) < batch_size ({batch_size}). Disabling drop_last.")
            s1_drop_last = False

        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, 
                                  shuffle=True, collate_fn=collate_func, drop_last=s1_drop_last, pin_memory=True)
        
        scheduler = CosineAnnealingWarmRestarts(optimizer, 2, 2, eta_min=int(self.min_lr), last_epoch=-1)
        
        print('Stage 1 training (Warmup)...')
        for epoch in range(1, self.max_epoch_stage1):
            train_loss = train2(self.model, optimizer, train_loader, epoch, self.device, loss_fn, scheduler=scheduler)
            train_acc, train_f1 = test2(self.model, train_loader, predicts=False, device=self.device)
            lr = optimizer.param_groups[0]['lr']
            print(f'epoch {epoch:03d}, lr: {lr:.06f}, loss: {train_loss:.06f}, T-acc: {train_acc:.04f}, T-f1: {train_f1:.04f}')

        # Stage 2 Training
        print('Stage 2 training (Fine-tuning)...')
        
        raw_dataset = ExprDataset(Expr=logExpr, edge=used_edge, y=label, device='cpu')
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        y_cpu = raw_dataset.y.numpy()
        train_idx, valid_idx = next(sss.split(y_cpu, y_cpu))
        
        st2_train_dataset = raw_dataset.split(t.tensor(train_idx).long())
        st2_valid_dataset = raw_dataset.split(t.tensor(valid_idx).long())
        
        st2_train_dataset.duplicate_minor_types(dup_odds=50)
        
        # 【修复】Stage 2 关键修复：动态检查 drop_last
        # 因为切分后数据变少了，更容易触发这个问题
        s2_drop_last = True
        if len(st2_train_dataset) < batch_size:
            print(f"Notice: Stage 2 train size ({len(st2_train_dataset)}) < batch_size ({batch_size}). Disabling drop_last.")
            s2_drop_last = False

        st2_train_loader = DataLoader(st2_train_dataset, batch_size=batch_size, num_workers=0, 
                                      shuffle=True, collate_fn=collate_func, drop_last=s2_drop_last, pin_memory=True)
        st2_valid_loader = DataLoader(st2_valid_dataset, batch_size=batch_size, num_workers=0, 
                                      shuffle=True, collate_fn=collate_func, pin_memory=True)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Stage 2 initialize lr: {current_lr}')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2, verbose=True, min_lr=0.00001)

        max_metric = 0.0
        max_metric_count = 0
        
        for epoch in range(self.max_epoch_stage1, self.max_epoch_stage1 + self.max_epoch_stage2):
            train_loss = train2(self.model, optimizer, st2_train_loader, epoch, self.device, loss_fn, verbose=False)
            train_acc, train_f1 = test2(self.model, st2_train_loader, predicts=False, device=self.device)
            valid_acc, valid_f1 = test2(self.model, st2_valid_loader, predicts=False, device=self.device)
            
            lr = optimizer.param_groups[0]['lr']
            print(f'epoch {epoch:03d}, lr: {lr:.06f}, loss: {train_loss:.06f}, T-acc: {train_acc:.04f}, T-f1: {train_f1:.04f}, V-acc: {valid_acc:.04f}, V-f1: {valid_f1:.04f}')
            
            scheduler.step(valid_f1)
            
            if valid_f1 > max_metric:
                max_metric = valid_f1
                max_metric_count = 0
                t.save(self.model.state_dict(), os.path.join(self.out_dir, 'models', 'best_model.pth'))
            else:
                max_metric_count += 1
                if max_metric_count > 3:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break
            
            if optimizer.param_groups[0]['lr'] <= 0.00001:
                print('Minimum learning rate reached.')
                break
        
        best_model_path = os.path.join(self.out_dir, 'models', 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(t.load(best_model_path))
            print("Loaded best model state from Stage 2.")

        t.save(self.model, os.path.join(self.out_dir, 'models', 'final_model.pth'))




    def _prepare_test_loader(self, dataset):
        if self.model is None:
             raise RuntimeError("Model is not fitted yet.")
        if isinstance(dataset, DataLoader):
            return dataset
        
        if hasattr(dataset, 'X'):
            expr = dataset.X
            y = np.zeros(expr.shape[0])
        elif isinstance(dataset, (np.ndarray, torch.Tensor)):
            expr = dataset
            y = np.zeros(expr.shape[0])
        elif hasattr(dataset, 'Expr'):
            expr = dataset.Expr
            y = dataset.y
        else:
            raise ValueError("Unsupported input type for prediction.")

        used_edge = self.dataset_attr.get('used_edge')
        if used_edge is None:
            raise RuntimeError("Missing edge info.")

        test_dataset = ExprDataset(Expr=expr, edge=used_edge, y=y, device='cpu')
        return DataLoader(test_dataset, batch_size=1, num_workers=0, collate_fn=collate_func)

    def predict_proba(self, dataset=None):
        test_loader = self._prepare_test_loader(dataset)
        _, _, _, _, probs, _ = test2(self.model, test_loader, predicts=True, device=self.device)
        return probs

    def predict(self, dataset=None):
        test_loader = self._prepare_test_loader(dataset)
        _, _, _, y_pred, _, _ = test2(self.model, test_loader, predicts=True, device=self.device)
        y_pred = np.reshape(y_pred, -1)
        if self.str_labels:
            return np.array([self.str_labels[x] for x in y_pred])
        return y_pred