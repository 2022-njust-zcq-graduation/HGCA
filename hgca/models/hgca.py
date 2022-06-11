import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import HGCN


class HGCA(nn.Module):
    def __init__(self, in_ft, out_ft, hid_ft, tm_hid_ft, P, tau=0.5):
        # 输入特征维度，表示特征维度，元路径聚合中间维度，损失计算中间维度，元路径个数，激活函数
        super(HGCA, self).__init__()
        self.hgcn = HGCN(in_ft, out_ft, hid_ft, P)

        self.tau = tau
        self.fc1 = nn.Linear(out_ft, tm_hid_ft)
        self.fc2 = nn.Linear(tm_hid_ft, out_ft)

    def forward(self, seq1, seq2, adjs1, adjs2):
        h_1 = self.hgcn(seq1, adjs1)        # 正样本节点特征表示
        h_2 = self.hgcn(seq2, adjs2)        # 负样本节点特征表示
        h_1 = torch.squeeze(h_1, dim=0)
        h_2 = torch.squeeze(h_2, dim=0)
        return self.loss(h_1, h_2)          # 计算损失

    def embed(self, seq, adjs):
        h = self.hgcn(seq, adjs)
        return h.detach()

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))          # 每个节点与同一视图内节点的相似性
        between_sim = f(self.sim(z1, z2))       # 每个节点与另一个视图间节点的相似性
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, mean=True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
