from .torch_nn import *
from timm.models.layers import DropPath



class GlobalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act='GraphLU', norm=None, bias=True):
        super(GlobalConv2d, self).__init__()
        self.nn = BasicConv([in_channels*3, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_k=torch.mean(x_j,dim=-1,keepdim=True)
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x, x_k, x_j], dim=1).reshape(b, 3 * c, n, _)
        return self.nn(x)





class LocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(LocalConv2d, self).__init__()
        dim = in_channels
        self.cnn_in = cnn_in = dim
        self.cnn_dim = cnn_dim = cnn_in * 2
        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, stride=1, padding=1, bias=False,groups=cnn_dim)
        self.act = nn.GELU()

    def forward(self, x, edge_index=None, y=None):
        x = self.conv1(x)
        x = self.proj1(x)
        x = self.act(x)
        return x




class GraphConv2d(nn.Module):
    """
    graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'global':
            self.gconv = GlobalConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'local':
            self.gconv = LocalConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index=None, y=None):
        return self.gconv(x, edge_index, y)


