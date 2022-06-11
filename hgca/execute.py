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


# 加载数据
adjs, features, labels, idx_train, idx_val, idx_test = load_data2(mask=[0.8, 0.1, 0.1])
features = preprocess_features(features)    # 节点特征归一化

P = int(len(adjs))              # 元路径个数
nb_nodes = features.shape[0]    # 节点数
ft_size = features.shape[1]     # 输入特征维度
nb_classes = labels.shape[1]    # 标签维度

features = torch.FloatTensor(features)
edges = []              # 每个同质图对应的边集
edges_data = []
drop_weights = []       # 每个同质图边的重要性
features_weights = torch.zeros(ft_size)     # 每个特征维度的重要性
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

# 删除部分边之后，将边集转换成邻接矩阵并归一化
def temp_adjs(temp_adjs):
    hh_adjs = []
    for temp_adj in temp_adjs:
        temp_adj = normalize_adj(temp_adj + sp.eye(temp_adj.shape[0]))
        temp_adj = (temp_adj + sp.eye(temp_adj.shape[0])).todense()
        temp_adj = temp_adj[np.newaxis]
        hh_adjs.append(temp_adj)
    return torch.FloatTensor(np.array(hh_adjs))


labels = torch.FloatTensor(labels[np.newaxis]).to(DEVICE)
idx_train = torch.LongTensor(idx_train).to(DEVICE)
idx_val = torch.LongTensor(idx_val).to(DEVICE)
idx_test = torch.LongTensor(idx_test).to(DEVICE)

# 模型定义
model = HGCA(ft_size, hid_units, shid, shhid, P, tau=0.4).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# 节点表示学习
patience = 30
wait = 0            # 最小损失后继续训练次数
best_loss = 1e9     # 最小损失
best_epoch = 0      # 最小损失的epoch

t_total = time.time()
f = open("HGCA.txt", 'w')
f.write("Optimization Beginning:\n")
print("Optimization Beginning:")
for epoch in range(EPOCHS):
    t = time.time()
    model.train()
    optimiser.zero_grad()

    features_1 = drop_feature_weighted_2(features, features_weights, 0.1)   # 掩盖部分维度
    features_2 = drop_feature_weighted_2(features, features_weights, 0.3)   # 掩盖部分维度
    features_1 = torch.unsqueeze(features_1, dim=0).to(DEVICE)
    features_2 = torch.unsqueeze(features_2, dim=0).to(DEVICE)
    # idx = np.random.permutation(nb_nodes)
    # features_2 = features_2[:, idx, :]

    adjs_1 = []         # 删除部分边
    adjs_2 = []         # 删除部分边
    for i in range(P):
        edge_1 = drop_edge_weighted(edges[i], drop_weights[i], p=0.1, threshold=0.7)
        edge_2 = drop_edge_weighted(edges[i], drop_weights[i], p=0.6, threshold=0.7)
        edge_1 = edge_to_adj2(edge_1, edges_data[i], nb_nodes)
        edge_2 = edge_to_adj2(edge_2, edges_data[i], nb_nodes)
        adjs_1.append(edge_1)
        adjs_2.append(edge_2)
    adjs_1 = temp_adjs(adjs_1).to(DEVICE)
    adjs_2 = temp_adjs(adjs_2).to(DEVICE)

    loss = model(features_1, features_2, adjs_1, adjs_2)

    if loss < best_loss:
        wait = 0
        best_loss = loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_hgca.pkl')
    else:
        wait += 1

    if wait == patience:
        f.write('Early stopping!\n')
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

    f.write("Epoch: {:04d}  Loss: {:.8f}  Time: {:.4f}s\n".format(epoch, loss.item(), time.time() - t))
    print("Epoch: {:04d}  Loss: {:.8f}  Time: {:.4f}s".format(epoch, loss.item(), time.time() - t))

f.write("Optimization Finished!\n")
f.write("Total time elapsed: {:.4f}s\n\n".format(time.time() - t_total))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))


# 用下游任务测试
f.write("Test Beginning:\n")
print("Test Beginning:")
f.write('Loading {}th epoch\n'.format(best_epoch))
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('best_hgca.pkl'))
embeds = model.embed(torch.unsqueeze(features, dim=0).to(DEVICE), temp_adjs(adjs).to(DEVICE))
train_embs = embeds[0, idx_train]   # 训练集节点特征表示
val_embs = embeds[0, idx_val]       # 验证集节点特征表示
test_embs = embeds[0, idx_test]     # 测试集节点特征表示

train_lbls = torch.argmax(labels[0, idx_train], dim=1)  # n * 1，转换成对应类别
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)


# 测试开始，分类过程10次取平均
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
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc)
    mac = torch.Tensor(np.array(macro_f1(preds, test_lbls)))
    mac_f1.append(mac)
    mic = torch.Tensor(np.array(micro_f1(preds, test_lbls)))
    mic_f1.append(mic)
    f.write("accuracy: {:.4f}  mic_f1: {:.4f}  mac_f1: {:.4f}  time: {:.4f}s\n".format(acc, mic, mac, time.time() - t))
    print("accuracy: {:.4f}  mic_f1: {:.4f}  mac_f1: {:.4f}  time: {:.4f}s".format(acc, mic, mac, time.time() - t))


# 输出平均结果
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
