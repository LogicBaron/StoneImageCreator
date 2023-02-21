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
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
     ])
    image_tensors = [jitter(image.convert("RGB")) for image in examples["image"]]
    return {'img_input': image_tensors}

def main():
    # load model
    
    model = UNet(in_dim=64,
                 #dim_mults = (1, 2, 2, 4, 8),
                 #is_attn = (False, False, False, False, True)
                 dim_mults = (1, 2, 4, 8, 16),
                 is_attn = (False, False, False, True, True)
                 )
    diffusion = DiffusionModel(model = model,
                               num_timesteps=1_000)
    # diffusion.load_state_dict(torch.load("best_model_399_.pt"))
    print(diffusion)
    # load data
    dataset = load_dataset("huggan/smithsonian_butterflies_subset")
    # dataset = load_dataset("tglcourse/CelebA-faces-cropped-128") #huggan/smithsonian_butterflies_subset
    # simple transform
    train_dataset = dataset['train']
    train_dataset.set_transform(transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128)
    # load optimizer, scheduler
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2_000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1_000)
    diffusion.to(torch.device("cuda:6"))

    best_loss = 100
    for epoch in range(5_000):
        # train
        print(f"{epoch}th epoch training...")
        loss_total = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            data = batch['img_input'].to("cuda:6")
            optimizer.zero_grad()
            loss = diffusion(data)
            loss.backward()
            optimizer.step()
            loss_total += loss
        train_avg_loss = loss_total/len(train_dataloader)
        print(f"train_loss: {train_avg_loss}, lr: {scheduler.get_last_lr()}")
        loss_total = 0
        scheduler.step()
        # eval
        if epoch < 1_000:
            if epoch % 50 == 0:
                with torch.no_grad():
                    x = diffusion.sample(16,3,128)
                imgs_grid = make_grid(x, 4, 4)
                imgs_grid.save(f"sample/sample_{epoch}.png")
                torch.save(diffusion.state_dict(), f"best_model.pt")
        else:
            if train_avg_loss < best_loss:
                best_loss = train_avg_loss
                with torch.no_grad():
                    x = diffusion.sample(16,3,128)
                imgs_grid = make_grid(x, 4, 4)
                imgs_grid.save(f"sample/sample_{epoch}.png")
                torch.save(diffusion.state_dict(), f"best_model.pt")
    

if __name__ == "__main__":
    main()