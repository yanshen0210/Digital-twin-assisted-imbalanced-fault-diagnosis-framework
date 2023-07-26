import torch
import torch.nn as nn
import numpy as np


class DWBLoss(nn.Module):
    def __init__(self, cls_num_list, reduction='mean'):
        super(DWBLoss, self).__init__()
        self.beta = np.log(cls_num_list.max() / cls_num_list) + 1
        self.beta = torch.cuda.FloatTensor(self.beta)
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        loss = self.beta[targets] ** (1 - pt) * CE_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
