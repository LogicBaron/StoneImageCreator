import torch
from torch import nn

class VaeBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim  = hidden_dim

        self.convs = nn.ModuleList([])
        in_c = in_dim
        out_c = hidden_dim

        for idx in range(n_layers):
            if idx == n_layers-1:
                out_c = out_dim
            self.convs.append(ConvBlock(in_c, out_c))
            in_c = out_c

        self.res = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1)) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor):
        h = x
        for conv in self.convs:
            h = conv(x)
        return self.res(x) + h

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__inti__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), padding=(1,1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        return self.conv(self.dropout(self.act(self.norm(x))))