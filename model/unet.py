import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, n_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        # Time embeding channel is 4 times of image dimension.
        self.time_dim = self.n_channels * 4

        downs = []


    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x : B x C x H x W
        t : B
        """

class ConvResBlock(nn.Module):
    """
    All models have two convolution residual blocks per resolution level.
    """
    def __init__(self, in_dim: int, out_dim: int, n_groups: int, dropout: float=0.1):
        """
        DDPM paper: We replaced weight normalization with group normalization
        n_grups: # of groups for group normalization
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_dim)
        self.act1 = nn.GeLU()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=(3,3), padding=(1,1))

        self.norm2 = nn.GroupNorm(n_groups, out_dim)
        self.act2 = nn.GeLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=(3,3), padding=(1,1))

        if in_dim != out_dim:
            self.res = nn.Conv2d(in_dim, out_dim, kernel_size=(1,1))
        else:
            self.res = nn.Identity()

        # Linear Layer for time embedding
        self.time_ln = nn.Linear(t_dim, out_dim)
        self.time_act = nn.GeLU()

        self.dropout = nn.dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x : `[B, in_dim, h, w]`
        t : `[B, time_dim]`
        """
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_ln(self.time_cat(t))[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.res(x)

class AttentionBlock(nn.Module):
    """
    All Models have self-attention blocks at the 16 x 16 resolution between the  convolutional blocks
    """

