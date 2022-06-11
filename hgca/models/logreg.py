import torch
import torch.nn as nn


# 下游节点分类任务，一个全连接层
class LogReg(nn.Module):
    def __init__(self, features, classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(features, classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
