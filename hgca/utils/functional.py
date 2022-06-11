import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import degree, to_undirected


# 邻接矩阵转换成边集
def adj_to_edge2(adj):
    edge_index_temp = sp.coo_matrix(adj.numpy())
    indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
    return torch.LongTensor(indices), dict(zip(zip(edge_index_temp.row, edge_index_temp.col), edge_index_temp.data))


# 边集转换成邻接矩阵
def edge_to_adj2(edge, c, n):
    row = edge[0].to('cpu').numpy()
    col = edge[1].to('cpu').numpy()
    data = dict([(key, c[key]) for key in list(zip(row, col))])
    adj = sp.coo_matrix((np.array(list(data.values())), (row, col)), shape=(n, n))
    return adj


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


# 按重要性掩盖节点部分维度
def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


# 维度重要性。节点特征:(N, F)，每个节点的度:(N)
def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
#    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


# 按重要性删除部分边
def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


# 边重要性。边:(2, E)
def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights
