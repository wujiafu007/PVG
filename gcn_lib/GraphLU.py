import torch
from torch import nn
import math


class GraphLU(nn.Module):
    def __init__(self):
        super(GraphLU, self).__init__()
        self.sigma = nn.Parameter(math.sqrt(2.0) * torch.ones(1), requires_grad=True)

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / (math.sqrt(2.0)+self.sigma)))