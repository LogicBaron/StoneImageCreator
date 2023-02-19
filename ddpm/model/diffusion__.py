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
        self.register_buffer('backward_process_coef1', 1. / torch.sqrt(self.alphas_cumprod))
        self.register_buffer('backward_process_coef2', torch.sqrt(1. / self.alphas_cumprod - 1))
        # coefficients for posteriors q(x_{t-1}|x_t, x_0).
        self.register_buffer('posterior_mean_coef1', torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(self.alphas_cumprod) * (1-self.alphas_cumprod_prev) / (1-self.alphas_cumprod))
        self.register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.register_buffer('posterior_log_variance', torch.log(self.posterior_variance.clamp(min=1e-20)))
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
    def sample(self, img_shape, batch_size=16, return_all_timesteps=False):
        print(f"Sampling {batch_size} images...")
        if self.sampling_timesteps < self.num_timesteps:
            return self.ddim_sample((batch_size,) + img_shape)
        else:
            return self.p_sample_loop((batch_size,) + img_shape)

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.model(xt, t)
        # [gather](utils.html) $\bar\alpha_t$

        alpha_bar = self.alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        # $\alpha_t$
        alpha = self.alphas.gather(-1, t).reshape(-1, 1, 1, 1)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = self.betas.gather(-1, t).reshape(-1, 1, 1, 1)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** .5) * eps   

    @torch.no_grad()
    def p_sample_(self, x: torch.Tensor, t: int):
        """
        backward pass
        """
        b, c, w, h  = x.shape
        device = x.device
        batched_t = torch.full((b,), t, device=device, dtype=torch.long)
        pred_noise = self.model(x, batched_t)
        pred_noise = pred_noise.clip(min=-1, max=1)
        self.backward_process_coef2.gather(-1, batched_t)
        x_start = self.backward_process_coef1.gather(-1, batched_t).view(b, 1, 1, 1) * x - \
                  self.backward_process_coef2.gather(-1, batched_t).view(b, 1, 1, 1) * pred_noise
        x_start = x_start.clip(min=-1, max=1)

        posterior_mean = self.posterior_mean_coef1.gather(-1, batched_t).view(b, 1, 1, 1) * x_start + \
                         self.posterior_mean_coef2.gather(-1, batched_t).view(b, 1, 1, 1) * x
        posterior_variance = self.posterior_variance.gather(-1, batched_t).view(b, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance.gather(-1, batched_t).view(b, 1, 1, 1)
        noise = torch.randn_like(x) if t > 1 else 0. # assume no noise if t==0 or 1.
        pred_img = posterior_mean + (0.5 * posterior_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device).clip(min=-1, max=1)
        imgs = [img]
        for t in tqdm(reversed(range(0, self.num_timesteps))):
            pred_img, x_start = self.p_sample(img, t)
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = unscale_img_linear(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        b, c, w, h  = shape
        device = self.betas.device
        times = torch.linspace(-1, self.num_timesteps-1, steps=self.sampling_timesteps+1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        img = torch.randn(shape, device=device).clip(min=-1, max=1)
        imgs = [img]
        for time, time_next in time_pairs:
            batched_t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise = self.model(img, batched_t)
            pred_noise = pred_noise.clip(min=-1, max=1)
            x_start = self.backward_process_coef1.gather(-1, batched_t).view(b, 1, 1, 1) * x - \
                    self.backward_process_coef2.gather(-1, batched_t).veiw(b, 1, 1, 1) * pred_noise
            x_start = x_start.clip(min=-1, max=1)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1-alpha_next) / (1-alpha)).sqrt()
            sigma = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.rand_like(img)

            img = x_start * alpha_next.sqrt() + sigma * pred_noise + sigma * noise
            imgs.append(img)
        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = unscale_img_linear(ret)
        return ret

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, c, w, h = x1.shape
        device = x1.device
        t = t if t is not None else self.num_timesteps-1

        assert x1.shape == x2.shape

        batched_t = torch.full((b,), t, device=device, dtype=torch.float)
        x1_start = self.q_sample(x, t=batched_t)
        x2_start = self.q_sample(x, t=batched_t)

        img = (1-lam) * x1_start + lam * x2_start

        for i in tqdm(reversed(range(0, t))):
            img, x_start = self.p_sample(img, i)
        return img

    def q_sample(self, x, t):
        """
        forward_process
        """
        noise = torch.randn_like(x)
        return self.forward_process_coef1.gather(-1, t).view(b, 1, 1, 1) * x + \
               self.forward_process_coef2.gather(-1, t).view(b, 1, 1, 1) * noise

if __name__ == '__main__':
    DiffusionModel()