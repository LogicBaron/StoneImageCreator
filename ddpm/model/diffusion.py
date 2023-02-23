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
        # loss
        self.loss_fn = F.mse_loss

        # p2 loss weight, from https://arxiv.org/abs/2204.00227
        p2_loss_weight_gamma = 0
        p2_loss_weight_k = 1
        self.register_buffer('p2_loss_weight',
            (p2_loss_weight_k + self.alphas_cumprod / (1 - self.alphas_cumprod)) ** -p2_loss_weight_gamma)

        # sampling
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else num_timesteps
        assert self.sampling_timesteps <= num_timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def forward(self, img):
        """
        Training Process.
        Implementation of `Algorithm1. Training`
        x : B x C x H x W
        """
        # image data scaled linearlly to [-1, 1] in ddpm.
        img = scale_img_linear(img)

        b, c, h, w = img.size()
        # t ~ Uniform({1,...,T}), noise ~ normal(0, I)
        # if t is 0, the diffused image is original image.
        # if t is 1, the diffused image is assumed to be noiseless at the end of sampling.
        t = torch.randint(0, self.num_timesteps, (b,), device = img.device).long()
        noise = torch.randn_like(img)

        x = self.q_sample(img, t, noise=noise)
        model_out = self.model(x, t)

        loss = self.loss_fn(model_out, noise, reduction = 'none')
        loss = self.p2_loss_weight.gather(-1, t).view(b,1,1,1) * loss

        return loss.mean()

    @torch.no_grad()
    def sample(self, n_samples, img_channels, img_size, noise_clamp=False, denoised_clamp=False):
        """
        Implementation of `Algorithm2. Sampling`
        """
        x = torch.randn([n_samples, img_channels, img_size, img_size], device=self.betas.device)
        for _, t_ in tqdm(enumerate(range(self.num_timesteps))):
            t = self.num_timesteps - t_ - 1
            x = self.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long),
                              noise_clamp=noise_clamp,
                              denoised_clamp=denoised_clamp)
        x = unscale_img_linear(x.clamp(min=-1, max=1))
        return x

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, noise_clamp=False, denoised_clamp=False):
        """
        rever process. remove noise from input.
        Implementation of  `Algorithm2. Sampling`'s inner loop.
        We use sigma as sqrt(betas).
        """
        pred_noise = self.model(x, t)
        if noise_clamp:
            pred_noise = pred_noise.clamp(min=-1, max=1)

        alphas_cumprod = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        alphas = self.alphas.gather(-1, t).reshape(-1, 1, 1, 1)
        noise_coef = (1-alphas) / torch.sqrt(1-alphas_cumprod)
        mean = 1 / alphas.sqrt() * (x - noise_coef * pred_noise)
        if denoised_clamp:
            mean = mean.clamp(min=-1, max=1)
        # Experimentally, both betas and betas^wave had similar results.
        # code uses betas as sigma square. 
        var = self.betas.gather(-1, t).view(-1, 1, 1, 1)

        noise = torch.randn(x.shape, device=x.device)

        return mean + (var**0.5)*noise

    @torch.no_grad()
    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise=None):
        """
        forward process or diffusion process. add Guassian noise to input.
        Implementation of q(x_t|x_0).
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        mean = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        mean = mean.sqrt() * x
        var = 1 - self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        return mean + (var ** 0.5) * noise

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, c, w, h = x1.shape
        device = x1.device
        t = t if t is not None else self.num_timesteps-1

        assert x1.shape == x2.shape

        batched_t = torch.full((b,), t, device=device, dtype=torch.long)
        x1 = scale_img_linear(x1)
        x2 = scale_img_linear(x2)
        x1_t = self.q_sample(x1, t=batched_t)
        x2_t = self.q_sample(x2, t=batched_t)
        x_t = (1-lam) * x1_t + lam * x2_t

        for _, t_ in tqdm(enumerate(range(t))):
            step = t - t_ - 1
            x_t = self.p_sample(x_t, x_t.new_full((b,), step, dtype=torch.long))
        x = unscale_img_linear(x_t.clamp(min=-1, max=1))
        return x
        
if __name__ == '__main__':
    DiffusionModel()