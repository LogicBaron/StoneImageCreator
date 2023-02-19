from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPImageProcessor
from torchvision.transforms import Compose, Normalize, ToTensor

from model import UNet, DiffusionModel

def transforms(examples):
    jitter = Compose([
        ToTensor(),
        Normalize(mean=[0.4913, 0.4821, 0.4465], std=[0.2470, 0.2434, 0.2615]),
     ])
    examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["img"]]
    return examples


def collate_fn(examples):
    images = []
    for example in examples:
        images.append((example['pixel_values']))
    images = torch.stack(images)
    return {'img_input': images}


def main():
    # load model
    model = UNet()
    ddpm = DiffusionModel(model = model,
                          num_timesteps=1000)
    print(ddpm)
    # load data
    dataset = load_dataset("cifar10")
    # simple transform
    train_dataset = dataset['train']
    train_dataset.set_transform(transforms)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=128)
    test_dataset = dataset['test'].with_transform(transforms)
    test_dataset.set_transform(transforms)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=128)

    # load optimizer, scheduler
    optimzer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    ddpm.to(torch.device("cuda:3"))
    
    for epoch in range(5):
        # train
        print(f"{epoch}th epoch training...")
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            loss = ddpm(batch['img_input'].to("cuda:3"))
            loss.backward()
            optimzer.step()
            optimzer.zero_grad()
            total_loss += loss
        print(f"train loss : {total_loss/len(train_dataloader)}")
        # eval
        with torch.no_grad():
            total_loss = 0
            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                total_loss += ddpm(batch["img_input"].to("cuda:3"))
            print(f"eval loss : {total_loss/len(test_dataloader)}")
    

if __name__ == "__main__":
    main()