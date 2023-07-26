import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MAR(nn.Module):

    def __init__(self, cls_num_list, weight=None, max_m=0.5, s=1):
        super(MAR, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        # if phase == 'source_train':
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index.type(torch.bool), x_m, x)
        loss = F.cross_entropy(self.s * output, target, weight=self.weight)

        # elif phase == 'target_test':
        #     loss = F.cross_entropy(x, target, weight=self.weight)

        return loss
