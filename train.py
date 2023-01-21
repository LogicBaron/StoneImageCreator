import torch
from model import UNet

def main():
    # load model
    model = UNet()
    ddpm = DiffusionModel(model = model,
                          num_timesteps=1000)
    
    # load data
    train_dataset = None
    train_dataloader = None
    test_dataset = None
    test_dataloader = None

    # load optimizer, scheduler
    optimzer = None
    scheduler = None

    for epoch in range(5):
        # train
        for batch_idx, batch in enumerate(train_dataloader):
            loss = ddpm(batch)
            loss.backward()
            optimzer.step()
            optimzer.zero_grad()
        # eval
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                loss = ddpm(batch)
    

if __name__ == "__main__":
    main()