from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPImageProcessor
from torchvision import transforms

from model import UNet, DiffusionModel
# from model.unet_ import UNet
# from model.diffusion_ import DenoiseDiffusion
from utils import make_grid

from labml import lab, tracker, experiment, monit

def transform(examples):
    jitter = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
     ])
    image_tensors = [jitter(image.convert("RGB")) for image in examples["image"]]
    return {'img_input': image_tensors}

def main():
    # load model
    
    model = UNet(in_dim=64,
                 #dim_mults = (1, 2, 2, 4, 8),
                 #is_attn = (False, False, False, False, True)
                 dim_mults = (1, 2, 2, 4, 4),
                 is_attn = (False, False, False, False, True)
                 )
    diffusion = DiffusionModel(model = model,
                          num_timesteps=1_000)

    print(diffusion)
    # load data
    dataset = load_dataset("huggan/smithsonian_butterflies_subset") # tglcourse/CelebA-faces-cropped-128
    # simple transform
    train_dataset = dataset['train']
    train_dataset.set_transform(transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    # load optimizer, scheduler
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    diffusion.to(torch.device("cuda:3"))


    for epoch in range(1000):
        # train
        print(f"{epoch}th epoch training...")
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            data = batch['img_input'].to("cuda:3")
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
        print(f"train_loss: {loss}, lr: {scheduler.get_lr()}")
        scheduler.step()
        # eval
        if epoch%50 == 49:
            with torch.no_grad():
                x = diffusion.sample(16,3,64)
            imgs_grid = make_grid(x, 4, 4)
            imgs_grid.save(f"sample/sample_{epoch}.png")
            torch.save(diffusion.state_dict(), f"best_model_{epoch}.pt")
    

if __name__ == "__main__":
    main()