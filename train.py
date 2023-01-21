import torch
from model import UNet

def main():
    model = UNet()
    ddpm = DiffusionModel(model = model,
                          num_timesteps=1000)

if __name__ == "__main__":
    main()