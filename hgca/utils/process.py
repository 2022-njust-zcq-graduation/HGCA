import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch
import torch.nn as nn
import random
from sklearn import metrics


# 类别转换成one-hot向量
def encode_onehot(labels):
    classes = set(labels)   # 去重
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


# 加载IMDB数据集
def load_data1(path="../data/IMDB/3-class/", dataset="IMDB", mask=[0.8, 0.1, 0.1]):
    print('Loading {} dataset...'.format(dataset))
    labels = []
    with open(path + 'movie_feature_vector_6334.pickle', 'rb') as f:
        features = pkl.load(f)
    f.close
    with open(path + 'index_label.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            labels.append(int(line[1]))
    f.close
    labels = encode_onehot(labels)
    features = sp.csr_matrix(features, dtype=np.float32)

    adjs = []
    for adj_name in ['movie_director_movie', 'movie_actor_movie', 'movie_keyword_movie']:
        with open(path + '{}_adj.pickle'.format(adj_name), 'rb') as f:
            adj = pkl.load(f)
        f.close
        adjs.append(adj)

    nodes = features.shape[0]
    original = range(nodes)
    idx_train = random.sample(original, int(nodes * mask[0]))
    original = list(set(original) ^ set(idx_train))
    idx_val = random.sample(original, int(nodes * mask[1]))
    idx_test = list(set(original) ^ set(idx_val))

    return adjs, features, labels, idx_train, idx_val, idx_test


# 加载ACM数据集
def load_data2(path="../data/ACM/", dataset="ACM", mask=[0.8, 0.1, 0.1]):
    print('Loading {} dataset...'.format(dataset))
    with open(path + 'features.pickle', 'rb') as f:
        features = pkl.load(f)
    f.close
    with open(path + 'adjacents.pickle', 'rb') as a:
        adjs = pkl.load(a)
    a.close
    with open(path + 'labels.pickle', 'rb') as f:
        labels = pkl.load(f)
    f.close

    labels = labels.numpy()
    features = sp.csr_matrix(features, dtype=np.float32)

    nodes = features.shape[0]
    original = range(nodes)
    idx_train = random.sample(original, int(nodes * mask[0]))
    original = list(set(original) ^ set(idx_train))
    idx_val = random.sample(original, int(nodes * mask[1]))
    idx_test = list(set(original) ^ set(idx_val))

    return adjs, features, labels, idx_train, idx_val, idx_test


# 节点特征归一化
def preprocess_features(features):
    """归一化特征矩阵，每个元素除以元素所在行的和"""
    row_sum = np.array(features.sum(1))     # 行求和  (N * 1)
    r_inv = np.power(row_sum, -1).flatten() # 取倒数  N
    r_inv[np.isinf(r_inv)] = 0.             # 无穷为0
    r_mat_inv = sp.diags(r_inv)             # 变成矩阵的对角线
    features = r_mat_inv.dot(features)      # 矩阵乘法
    return features.todense()


# 邻接矩阵归一化
def normalize_adj(adj):
    """归一化邻接矩阵，D^(-1/2) * A * D^(-1/2)"""
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))                      # 行求和  (N * 1)
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()      # 取-1/2次方  N
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.               # 无穷为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)               # 变成矩阵的对角线
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def micro_f1(predicts, labels):
#     # Compute predictions
#     predicts = torch.round(nn.Sigmoid()(predicts))  # 变小数
#
#     # Cast to avoid trouble
#     predicts = predicts.long()
#     labels = labels.long()
#
#     # Count true positives, true negatives, false positives, false negatives
#     tp = torch.nonzero(predicts * labels).shape[0] * 1.0
#     tn = torch.nonzero((predicts - 1) * (labels - 1)).shape[0] * 1.0
#     fp = torch.nonzero(predicts * (labels - 1)).shape[0] * 1.0
#     fn = torch.nonzero((predicts - 1) * labels).shape[0] * 1.0
#
#     # Compute micro-f1 score
#     pre = tp / (tp + fp)
#     rec = tp / (tp + fn)
#     f1 = (2 * pre * rec) / (pre + rec)
#     return f1


# micro-f1评分标准
def micro_f1(predicts, labels):
    labels = labels.to(torch.device("cpu")).numpy()
    predicts = predicts.to(torch.device("cpu")).numpy()
    micro = metrics.f1_score(labels, predicts, average='micro')
    return micro


# macro-f1评分标准
def macro_f1(predicts, labels):
    labels = labels.to(torch.device("cpu")).numpy()
    predicts = predicts.to(torch.device("cpu")).numpy()
    macro = metrics.f1_score(labels, predicts, average='macro')
    return macro