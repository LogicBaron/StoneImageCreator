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
As the training pro
gresses, you can see that the following images are generated according to the Step.
``` shell
# train
>> python train.py
```
<img width="1042" alt="butterfly_images" src="https://user-images.githubusercontent.com/59866074/220822547-bc5b9ded-dac5-4ee8-8f66-ceef492a6f0c.png">

![sample_78](https://user-images.githubusercontent.com/59866074/220822761-10c85c67-b2f2-4aff-afea-1a5decd94cc8.png)


```python3
# load model
diffusion.load_state_dict(torch.load("path_to_best_model.pt"))

# sample
diffusion.sample(num_samples=16,
                 img_channels=3,
                 img_size=64)
```


## Interpolation
See `Interpolation.ipynb` notebook.

The lower the image, the deeper the t is.

![butterfly_interpolation_0](https://user-images.githubusercontent.com/59866074/220822652-c522b67c-a26f-4dca-9057-1d93f9ae3d8c.png)
---
![butterfly_interpolation_2](https://user-images.githubusercontent.com/59866074/220822657-0434dbfa-e7be-41df-89d0-9dcca0345c44.png)
---
![celeb_interpolation_0](https://user-images.githubusercontent.com/59866074/220822659-55dbd470-82aa-4190-b06c-e6ad5c5aa6a0.png)

