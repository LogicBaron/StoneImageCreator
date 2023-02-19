# [DDPM] Denoising Diffusion Probabilistic Models

For developers aiming to implement the paper, we wrote it based on the contents of the original paper as much as possible.
arxiv : https://arxiv.org/abs/2006.11239

## Tutorial

Base Load Model & Sampling
```python3
from model import UNet, DiffusionModel

# load model
model = UNet(in_dim=64,
             #dim_mults = (1, 2, 2, 4, 8),
             #is_attn = (False, False, False, False, True)
             dim_mults = (1, 2, 2, 4, 4),
             is_attn = (False, False, False, False, True)
             )
diffusion = DiffusionModel(model = model,
                           num_timesteps=1_000)
                           
                     
diffusion.load(""" pretrained_model_path """)

# sample
diffusion.sample(num_samples=16,
                 img_channels=3,
                 img_size=64)
```

## Train and Sampling
As the training progresses, you can see that the following images are generated according to the Step.

## Interpolation
