import math
import torch
from torch import nn
import torch.nn.functional as F

from model import UNet

def scale_img_linear(img):
    return img * 2 - 1

def unscale_img_linear(img):
    return (img + 1) / 2

def schedule_beta_linear(num_timesteps):
    scale = 1000/num_timesteps
    beta_start = scale * 1e-4
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_timestped, dtype=torch.float64)

class DiffusionModel():
    def __init__(self, 
                 model,
                 num_timesteps):
        self.model = model
        self.loss = F.mse_loss

        # forward process variances to constants increasing linearly from 1e-4 ~ 0.02.
        betas = schedule_beta_linear(num_timesteps)
        alpha = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.)

        # coefficients for closed form forward process q(x_t | x_0).
        forward_process_coef1 = torch.sqrt(alphas_cumprod)
        forward_process_coef2 = torch.sqrt(1. - alphas_cumprod)

        # coefficients for posteriors q(x_{t-1}|x_t, x_0).
        posterior_mean_coef1 = torch.sqrt(alphas_cumprod_prev) * betas / (1. - alphas_cumprod)
        posterior_mean_coef2 = torch.sqrt(alphas_cumprod) * (1-alphas_cumprod_prev) / (1-alphas_cumprod) 
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def forward(img):
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
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        epsilon = torch.randn_like(img)
        
        x = forward_process_coef1.gather(-1, t).view(b, 1, 1, 1) * img + \
            forward_process_coef2.gather(-1, t).view(b, 1, 1, 1) * epsilon

        model_out = self.model(x)

        loss = self.loss_fn(model_out, epsilon, reduction = 'none')

        return loss.mean()

    def q_sample():
        """
        backward process of remove noise.
        """

        # unscale image data.
        img = unscale_img_linear(img)

        return x

if __name__ == '__main__':
    DiffusionModel()