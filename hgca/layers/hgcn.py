import torch
import torch.nn as nn
from layers import GCN, SemanticAttentionLayer


# 组合多个图卷积层
class HGCN(nn.Module):
    def __init__(self, in_ft, out_ft, hid_ft, P):
        # 输入特征维度，表示特征维度，元路径聚合中间维度，元路径个数，激活函数
        super(HGCN, self).__init__()
        self.P = P
        self.gcn_level_embeddings = []
        for _ in range(P):
            self.gcn_level_embeddings.append(GCN(in_ft, out_ft, bias=True))

        for i, gcn_embedding_path in enumerate(self.gcn_level_embeddings):
            self.add_module('gcn_path_{}'.format(i), gcn_embedding_path)

        self.semantic_level_attention = SemanticAttentionLayer(out_ft, hid_ft)

    def forward(self, seq, adjacents):
        meta_path_x = []
        for i, adj in enumerate(adjacents):                         # 每个元路径对应的邻接矩阵
            m_x = self.gcn_level_embeddings[i](seq, adj)            # 节点对应gcn层表示特征
            meta_path_x.append(m_x)

        out = torch.cat([m_x for m_x in meta_path_x], dim=0)        # 各个元路径节点表示特征拼接
        out = self.semantic_level_attention(out, self.P)            # 节点表示特征聚合
        out = torch.unsqueeze(out, 0)
        return out
