import torch
import torch.nn as nn
import torch.nn.functional as F


# 语义级别attention，聚合基于特定元路径的节点表示以获得节点一般表示
class SemanticAttentionLayer(nn.Module):
    def __init__(self, out_ft, hid_ft):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = out_ft
        self.out_features = hid_ft

        self.W = nn.Parameter(torch.zeros(size=(out_ft, hid_ft)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, hid_ft)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, hid_ft)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)

        self.Tanh = nn.Tanh()

    def forward(self, inputs, P):
        # 各个元路径节点表示特征的拼接，元路径个数
        # input: (PN) * F
        # q.T * tanh(W * h + b)
        h = torch.mm(inputs, self.W)                                # (PN) * F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))      # (PN) * F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)    # P * N

        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)         # P * 1，取平均
        semantic_attentions = F.softmax(semantic_attentions, dim=0)                 # P * 1，各个元路径权重
        semantic_attentions = semantic_attentions.view(P, 1, 1)                     # P * 1 * 1，改变维度
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features)    # P * N * F

        # input_embedding: P * N * F
        input_embedding = inputs.view(P, N, self.in_features)

        # h_embedding: N * F
        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding
