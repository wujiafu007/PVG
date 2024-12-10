# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
from torch import nn
import torch.nn.functional as F


def xy_euclidean_distance(x, y):
    with torch.no_grad():
        xy_inner = -2*torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)
        # return torch.matmul(x, y.transpose(2, 1))



def knn_edge_select(x, y, k=9):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_euclidean_distance(x.detach(), y.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)




class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k

    def forward(self, x, y=None, relative_pos=None):
        x = F.normalize(x, p=2.0, dim=1)
        y = F.normalize(y, p=2.0, dim=1)
        edge_index = knn_edge_select(x, y, self.k * self.dilation)
        edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index
