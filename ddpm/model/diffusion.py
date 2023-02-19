from tqdm import tqdm

import math
import torch
from torch import nn
import torch.nn.functional as F

from .unet import UNet

def scale_img_linear(img):
    return img * 2 - 1

def unscale_img_linear(img):
    return (img + 1) / 2

def schedule_beta_linear(num_timesteps):
    scale = 1000/num_timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float)

class DiffusionModel(nn.Module):
    def __init__(self, 
                 model,
                 num_timesteps,
                 sampling_timesteps=None,
                 ddim_sampling_eta = 0.,):
        super().__init__()
        self.model = model
        self.loss = F.mse_loss
        self.num_timesteps = num_timesteps

        # forward process variances to constants increasing linearly from 1e-4 ~ 0.02.
        self.register_buffer('betas', schedule_beta_linear(self.num_timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1,0), value=1.))
        # coefficients for closed form forward process q(x_t | x_0).
        self.register_buffer('forward_process_coef1', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('forward_process_coef2', torch.sqrt(1. - self.alphas_cumprod))
        # coefficients for closed form backward process p(x_t-1 | x_t)
        # loss
        self.loss_fn = F.mse_loss
        # loss weight, from https://arxiv.org/abs/2204.00227
        self.p2_loss_weight_gamma = 0
        self.p2_loss_weight_k = 1
        self.register_buffer('p2_loss_weight', (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod))

        # sampling
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else num_timesteps
        assert self.sampling_timesteps <= num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def forward(self, img):
        """
        diffusion process.
        x : B x C x H x W
        """
        # image data scaled linearlly to [-1, 1] in ddpm.
        img = scale_img_linear(img)

        b, c, h, w = img.size()
        # t ~ Uniform({1,...,T}), epsilon ~ normal(0, I)
        # if t is 0, the diffused image is original image.
        # if t is 1, the diffused image is assumed to be noiseless at the end of sampling.
        t = torch.randint(0, self.num_timesteps, (b,), device = img.device).long()
        epsilon = torch.randn_like(img)

        x = self.forward_process_coef1.gather(-1, t).view(b, 1, 1, 1) * img + \
            self.forward_process_coef2.gather(-1, t).view(b, 1, 1, 1) * epsilon
        model_out = self.model(x, t)

        loss = self.loss_fn(model_out, epsilon, reduction = 'none')
        loss = self.p2_loss_weight.gather(-1, t).view(b,1,1,1) * loss

        return loss.mean()

    @torch.no_grad()
    def sample(self, n_samples, img_channels, img_size):
        x = torch.randn([n_samples, img_channels, img_size, img_size], device=self.betas.device)
        for _, t_ in tqdm(enumerate(range(self.num_timesteps))):
            t = self.num_timesteps - t_ - 1
            x = self.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor):
        pred_noise = self.model(x, t)

        alphas_cumprod = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas = self.alphas.gather(-1, t).reshape(-1, 1, 1, 1)
        noise_coef = (1-alphas) / torch.sqrt(1-alphas_cumprod)
        mean = 1 / alphas.sqrt() * (x - noise_coef * pred_noise)
        var = self.betas.gather(-1, t).view(-1, 1, 1, 1)

        noise = torch.randn(x.shape, device=x.device)
        return mean + (var**0.5)*noise

    @torch.no_grad()
    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(x)
        
        mean, var = self.q_xt_x0(x, t)
        return mean + (var ** 0.5) * noise

    @torch.no_grad()
    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor):
        mean = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        mean = mean.sqrt() * x
        var = 1 - self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        return mean, var

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, c, w, h = x1.shape
        device = x1.device
        t = t if t is not None else self.num_timesteps-1

        assert x1.shape == x2.shape

        batched_t = torch.full((b,), t, device=device, dtype=torch.float)
        x1_start = self.q_sample(x, t=batched_t)
        x2_start = self.q_sample(x, t=batched_t)

        x = (1-lam) * x1_start + lam * x2_start
        for _, t_ in tqdm(enumerate(range(t))):
            t = self.num_timesteps - t_ - 1
            x = self.p_sample(x, x.new_full((n_samples,), t, dtype=torch.lonwwg))
        return x
        
if __name__ == '__main__':
    DiffusionModel()