import torch
import torch.nn as nn


# 图卷积层
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)      # 权重维度
        self.act = nn.PReLU()                               # 激活函数

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        # 输入特征 * 权重
        out = self.fc(seq)
        # 邻接矩阵 * 输入特征 * 权重
        out = torch.bmm(adj, out)
        if self.bias is not None:
            out += self.bias
        # 激活函数
        out = torch.squeeze(self.act(out), 0)
        return out
