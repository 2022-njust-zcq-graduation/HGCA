import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected

from models import HGCA, LogReg
from utils.process import load_data1, load_data2, preprocess_features, normalize_adj, macro_f1, micro_f1
from utils.functional import adj_to_edge2, degree_drop_weights, feature_drop_weights_dense, drop_feature_weighted_2, \
                        drop_edge_weighted, edge_to_adj2


# 超参数定义
EPOCHS = 1000
lr = 0.001              # 学习率
weight_decay = 0.0      # 权重衰减
hid_units = 512         # 表示特征维度
shid = 16               # 元路径聚合中间维度
shhid = 64              # 损失计算中间维度
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


adjs, features, labels, idx_train, idx_val, idx_test = load_data2(mask=[0.8, 0.1, 0.1])
features = preprocess_features(features)    # 节点特征归一化

P = int(len(adjs))              # 元路径个数
nb_nodes = features.shape[0]    # 节点数
ft_size = features.shape[1]     # 输入特征维度
nb_classes = labels.shape[1]    # 标签维度

features = torch.FloatTensor(features)
edges = []
edges_data = []
drop_weights = []
features_weights = torch.zeros(ft_size)
for adj in adjs:
    adj = torch.FloatTensor(adj.toarray())
    edge, edge_data = adj_to_edge2(adj)
    b = degree_drop_weights(edge)
    drop_weights.append(b)
    edges.append(edge)
    edges_data.append(edge_data)

    edge_index_ = to_undirected(edge)
    node_deg = degree(edge_index_[1])
    node_deg = torch.sum(adj, dim=1)
    feature_weights = feature_drop_weights_dense(features, node_c=node_deg).div(P)
    features_weights.add_(feature_weights)


def temp_adjs(temp_adjs):
    hh_adjs = []
    for temp_adj in temp_adjs:
        temp_adj = normalize_adj(temp_adj + sp.eye(temp_adj.shape[0]))      # 矩阵对角线加1，并归一化
        temp_adj = (temp_adj + sp.eye(temp_adj.shape[0])).todense()
        temp_adj = temp_adj[np.newaxis]
        hh_adjs.append(temp_adj)
    return torch.FloatTensor(np.array(hh_adjs))


labels = torch.FloatTensor(labels[np.newaxis]).to(DEVICE)
idx_train = torch.LongTensor(idx_train).to(DEVICE)
idx_val = torch.LongTensor(idx_val).to(DEVICE)
idx_test = torch.LongTensor(idx_test).to(DEVICE)

model = HGCA(ft_size, hid_units, shid, shhid, P, tau=0.4).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


f = open("./temp/HGCA.txt", 'w')
f.write("Test Beginning:\n")
print("Test Beginning:")
f.write('Loading best epoch\n')
print('Loading best epoch')
model.load_state_dict(torch.load("./temp/ACM0.8.pkl", map_location=DEVICE))
embeds = model.embed(torch.unsqueeze(features, dim=0).to(DEVICE), temp_adjs(adjs).to(DEVICE))
train_embs = embeds[0, idx_train]   # 训练集节点特征表示
val_embs = embeds[0, idx_val]       # 验证集节点特征表示
test_embs = embeds[0, idx_test]     # 测试集节点特征表示

train_lbls = torch.argmax(labels[0, idx_train], dim=1)  # n * 1，转换成对应类别
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)


accs = []
mac_f1 = []
mic_f1 = []
for _ in range(10):
    t = time.time()
    bad_counter = 0
    patience = 20
    best = 10000
    best_epoch = 0
    loss_values = []

    log = LogReg(hid_units, nb_classes).to(DEVICE)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(10000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = loss_func(logits, train_lbls)
        logits_val = log(val_embs)
        loss_val = loss_func(logits_val, val_lbls)
        loss_values.append(loss_val)

        loss.backward()
        opt.step()

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            torch.save(log.state_dict(), 'best_test.pkl')
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

    # Restore best model
    f.write('Loading {}th epoch\n'.format(best_epoch))
    print('Loading {}th epoch'.format(best_epoch))
    log.load_state_dict(torch.load('best_test.pkl'))

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)     # N, 不是one-hot向量
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc)
    mac = torch.Tensor(np.array(macro_f1(preds, test_lbls)))
    mac_f1.append(mac)
    mic = torch.Tensor(np.array(micro_f1(preds, test_lbls)))
    mic_f1.append(mic)
    f.write("accuracy: {:.4f}  mic_f1: {:.4f}  mac_f1: {:.4f}  time: {:.4f}s\n".format(acc, mic, mac, time.time() - t))
    print("accuracy: {:.4f}  mic_f1: {:.4f}  mac_f1: {:.4f}  time: {:.4f}s".format(acc, mic, mac, time.time() - t))

accs = torch.stack(accs)
f.write('Average accuracy: {}\n'.format(accs.mean()))
print('Average accuracy:', accs.mean())

mic_f1 = torch.stack(mic_f1)
f.write('Average mic_f1: {}\n'.format(mic_f1.mean()))
print('Average mic_f1:', mic_f1.mean())

mac_f1 = torch.stack(mac_f1)
f.write('Average mac_f1: {}\n'.format(mac_f1.mean()))
print('Average mac_f1:', mac_f1.mean())
f.close()
