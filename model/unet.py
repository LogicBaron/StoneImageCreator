"""
top-down 
"""
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, 
                 in_dim: int = 64,
                 out_dim: int = None,
                 dim_mults = (1, 2, 4, 8),
                 is_attn = (False, False, True, True),
                 self_condition = Flase):
        super().__init__()
        self.n_resolutions = len(dim_mults)
        dim_mults = [*map(lambda x: in_dim * x, dim_mults)]
        in_out = list(zip(dim_mults[:-1], dim_mults[1:]))

        # Time embeding channel is 4 times of image dimension.
        self.time_dim = self.n_channels * 4

        # initial projection
        image_channel = 3 * (2 if self_condition else 1)
        self.init_conv = nn.Conv2d(image_channel, in_dim, kernel_size = 3, padding = 1)

        # time projection
        self.time_emb = TimeEmbedding(in_dim * 4)

        # down-sampling module list
        self.downs = nn.ModuleList([])
        for idx, (dim1, dim2) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(DownBlock(dim1, dim2, time_dim=time_dim, has_attn=is_attn[i]))
            self.downs.append(DownBlock(dim2, dim2, time_dim=time_dim, has_attn=is_attn[i]))
            if idx < num_resolutions - 1:
                self.downs.append(DownSample(dim2, dim2))



    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x : B x C x H x W
        t : B
        """
        pass

class DownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, time_dim: int, n_groups: int, has_attn: bool):
        super().__init__()
        self.convres = ConvResBlock(in_dim, out_dim:, time_dim, n_groups)
        if has_attn:
            self.attn = AttentionBlock(out_dim)
        else:
            self.attn = nn.Identity
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.convres(x, t)
        x = self.attn(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # use strides instead of max-pooling.
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        return self.conv(x)


class ConvResBlock(nn.Module):
    """
    All models have two convolution residual blocks per resolution level.
    """
    def __init__(self, in_dim: int, out_dim: int, n_groups: int = 32, dropout: float = 0.1):
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
    def __init__(self, in_dim, n_heads: int = 4, dim_head: int = 32, n_groups: int = 32):
        super().__Init__()
        self.scale = dim_head ** -0.5
        self.n_heads = n_heads
        self.dim_head = dim_head
        # layers
        self.norm = nn.GroupNorm(n_groups, in_dim)
        self.to_qkv = nn.Linear(in_dim, n_heads * dim_head * 3)
        self.to_out = nn.Linear(n_heads * dim_head, in_dim)

    def forward(self, x):
        """
        x : `[B, in_dim, h, w]`
        """
        b, c, h, w = x.shape
        h = self.norm(x) # x: `[b, c, h, w]`
        h = h.view(b, c, -1).permute(0, 2, 1) # x : `[b, h+w, c]`
        q, k, v = self.to_qkv(h).chunk(3, dim = 1) # q, k, v: `[b, h+w, n_heads * dim_head]`
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale # `[b, h+w, h+w]`
        attn = attn.softmax(dim=2) # `[b, h+w, h+w]`
        out = torch.matmul(attn, v) # `[b, h+w, n_heads * dim_head]`
        out = self.to_out(out) # `[b, h+w, c]`
        out = out.permute(0, 2, 1).view(b, c, h, w) # out: `[b, c, h, w]`

        return out + x

class TimeEmbedding(nn.Module):
    """
    Diffusion time t is specified by adding the Transformer sinusoidal position embedding
    """
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.ln1 = nn.Linear(self.time_dim // 4, self.time_dim)
        self.act1 = nn.GeLU()
        self.ln2 = nn.Linear(self.time_dim, self.time_dim)
    
    def forward(self, t: torch.Tensor):
        """
        t: `[B]`
        """
        # time sinusoidal osition embedding.
        half_dim = self.time_dim // 8
        time_emb = math.log(10_000) / (half_dim-1)
        time_emb = torch.exp(torch.arange(half_dim, device=t.device) * -time_emb)
        time_emb = t[:, None] * time_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=1)
        # MLP.
        time_emb = self.ln2(self.act1(self.ln1(time_emb)))
        return time_emb


