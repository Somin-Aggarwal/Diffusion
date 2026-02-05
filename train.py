import torch
import torch.nn as nn
from torch.optim import AdamW,lr_scheduler
import os 
from dataloader import DiffusionDataset
from torch.utils.data import DataLoader
from my_model import UNet, UNet_conditional
import argparse
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from my_utils import save_checkpoint, label_converter
import random
import numpy as np

def train(config, resume, weights_path):
    os.makedirs(config["logging"]["weights_dir"], exist_ok=True)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Dataset
    train_dataset = DiffusionDataset(
        file_path=config["train_file_path"],
        mode="train",
        steps=config["steps"],
        schedule=config["schedule"],
    )
    val_dataset = DiffusionDataset(
        file_path=config["val_file_path"],
        mode="test",
        steps=config["steps"],
        schedule=config["schedule"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        num_workers=config["num_workers"],
        pin_memory=config["device"].startswith("cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["device"].startswith("cuda"),
    )

    # Model
    mcfg = config["model"]
    model = UNet_conditional(
        img_ch=mcfg["img_ch"],
        base_ch=mcfg["base_ch"],
        ch_mul=mcfg["ch_mul"],
        attn=mcfg["attn"],
        n_resblocks=mcfg["n_resblocks"],
        tdim=mcfg["tdim"],
        steps=config['steps'],
        n_classes=config["n_classes"]
    ).to(config["device"])

    optimizer = AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    epochs = config["training"]["epochs"]
    warmup_epochs = min(config["training"]["warmup_epochs"], epochs - 1)

    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=config["training"]["eta_min"],
    )
    
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    start_epoch = 0
    if resume:
        weights_dict = torch.load(weights_path, map_location=config['device'])
        model.load_state_dict(weights_dict['model_state_dict'])
        optimizer.load_state_dict(weights_dict['optimizer_state_dict'])
        scheduler.load_state_dict(weights_dict['scheduler_state_dict'])
        start_epoch = weights_dict['epoch'] + 1

    best_val_loss = float("inf")
    null_idx = config["n_classes"] # idx

    for epoch in tqdm(range(start_epoch,epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for i, (noisy_image, actual_noise, time, labels) in enumerate(train_iterator):
            noisy_image = noisy_image.to(config["device"])
            actual_noise = actual_noise.to(config["device"])
            time = time.to(config["device"])
            labels = labels.to(config["device"])

            train_labels = label_converter(labels=labels,
                                           null_idx=null_idx,
                                           unconditional_percentage=config["uncond_per"],
                                           batch_size=config["training"]["batch_size"],
                                           device=config["device"])
            
            
            optimizer.zero_grad()
            predicted_noise = model(noisy_image, time, train_labels)
            loss = criterion(predicted_noise, actual_noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator.set_postfix(loss=loss.item(), avg_loss=epoch_loss/(i+1), lr=current_lr)

        scheduler.step()

        # Validation
        if (epoch + 1) % config["logging"]["validate_every_epoch"] == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for noisy_image, actual_noise, time, labels in val_loader:
                    noisy_image = noisy_image.to(config["device"])
                    actual_noise = actual_noise.to(config["device"])
                    time = time.to(config["device"])
                    labels = labels.to(config["device"])

                    pred = model(noisy_image, time, labels)
                    val_loss += criterion(pred, actual_noise).item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    "best_model.pt",
                    model.state_dict(),
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    epoch,
                    i,
                    val_loss,
                    config,
                    config["logging"]["weights_dir"],
                )
                print(f"Val loss : {val_loss} | BEST MODEL")
            else:
                print(f"Val loss : {val_loss}")

        # Epoch checkpoint
        if (epoch + 1) % config["logging"]["save_every_n_epochs"] == 0:
            save_checkpoint(
                f"epoch{epoch+1}.pt",
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                epoch,
                i,
                epoch_loss / len(train_loader),
                config,
                config["logging"]["weights_dir"],
            )
           

if __name__ == "__main__":
    
    resume = False
    weights_path = None
    
    if resume:
        config = torch.load(weights_path)['training_config']
    else:
        config = {
            "train_file_path": "cifar10_data.pkl",
            "val_file_path": "cifar10_data.pkl",
            "num_workers": 12,
            "schedule": "cosine",
            "steps": 1000,
            "n_classes": 10,
            "uncond_per" : 0.15,     # compute f(t)


            "model": {
                "img_ch" : 3,
                "base_ch": 128,
                "ch_mul": [1, 2, 2, 2],
                "attn": [1],
                "n_resblocks": 2,
                "tdim": 512,
            },

            "training": {
                "learning_rate": 2e-4,
                "epochs": 800,
                "batch_size": 8,
                "shuffle": True,
                "warmup_epochs": 20,
                "eta_min": 1e-5,
            },

            "logging": {
                "save_every_n_epochs": 100,
                "validate_every_epoch": 40,
                "weights_dir": "cifar_clsfree",
            },

            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "seed": 42,
        }
    
    print(config)
    train(config, resume, weights_path)