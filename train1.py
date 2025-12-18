import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
import os 
from dataloader import MNISTdatasetDiffusion
from model import UNET, UNET_old
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import logging
import torch.optim.lr_scheduler as lr_scheduler
from argparse import Namespace
import sys
from my_utils import save_checkpoint
import random
import numpy as np

logger = logging.getLogger(__name__)

def train(args, config=None, resume=False):
    
    os.makedirs(args.weights_dir, exist_ok=True)

    logging.basicConfig(
    filename=f"{args.weights_dir}/{args.log_file}",
    filemode='a',  # Append mode; creates file if it doesn't exist
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)   
    logger.info("\n\n\nSTARTED TRAINING\n")
    print(vars(args))
    logger.info(f"{vars(args)}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    # Load datasets
    train_dataset = MNISTdatasetDiffusion(file_path=args.train_file_path, mode="train", steps=args.steps)
    val_dataset = MNISTdatasetDiffusion(file_path=args.val_file_path, mode="test", steps=args.steps)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNET(image_channels=args.img_ch,
                 time_dim=args.time_dim).to(args.device)

    criterion = nn.MSELoss()
    optimizer = AdamW(params=model.parameters(),lr=args.learning_rate, betas=(args.beta1,args.beta2), eps=args.eps)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)
    
    start_epoch = 0
    skip_iteration = -1
    if resume and os.path.exists(f"{args.weights_dir}/best_model.pt"):
        best_val_loss = torch.load(f"{args.weights_dir}/best_model.pt")['loss']
    else:
        best_val_loss = 1e10
    
    # Resume from checkpoint
    if resume and config is not None:
        # Load saved states
        start_epoch = config['epoch']
        skip_iteration = config['iteration']
        model.load_state_dict(config['model_state_dict'])
        optimizer.load_state_dict(config['optimizer_state_dict'])
        if 'scheduler_state_dict' in config:
            scheduler.load_state_dict(config['scheduler_state_dict'])
        print(f"Resuming from epoch {start_epoch}, iteration {skip_iteration}")
        logging.info(f"Resumed: epoch {start_epoch}, iter {skip_iteration}, best_val_loss {best_val_loss}")
    
    epochs = args.epochs
    total_batches = len(train_loader)
    # if save_every_n_iteration_percentage = 5 then 100/5 = 20 files will be stored for every iteration
    save_every = max(1, total_batches * args.save_every_n_iteration_percentage // 100)
    validate_every = max(1, total_batches * args.validation_every_n_iteration_percentage // 100)
    
    print(f"Total batches: {total_batches}, save every {save_every}, validate every {validate_every}")
    logging.info(f"Total batches: {total_batches}, save_every: {save_every}, validate_every: {validate_every}")
    
    for current_epoch in tqdm(range(start_epoch,epochs),desc="Epochs"):
        
        epoch_loss = 0.0
        train_iterator = tqdm(train_loader, desc=f"Training Epoch {current_epoch+1}", leave=False)
        
        for current_iteration, batch in enumerate(train_iterator):
            # skip already processed iterations
            if current_epoch == start_epoch and current_iteration <= skip_iteration:
                continue

            noisy_image, actual_noise , time, label = batch

            noisy_image = noisy_image.to(args.device)
            actual_noise = actual_noise.to(args.device)
            time = time.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()
            
            predicted_noise = model(noisy_image, time)

            loss = criterion(predicted_noise, actual_noise)   
                        
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item() 
            epoch_loss += loss_value

            # Display iteration loss on tqdm bar
            train_iterator.set_postfix(iter_loss=loss_value)
            logger.info(f"Epoch : {current_epoch} | Iteration : {current_iteration} | Iteration Loss : {loss_value} | Running Epoch Loss : {epoch_loss}")
            
            # Save intermediate checkpoint
            if (current_iteration + 1) % save_every == 0:
                save_checkpoint(
                    f"checkpoint_epoch{current_epoch}_iter{current_iteration}.pt",
                    model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                    current_epoch, current_iteration, loss_value,
                    args, args.weights_dir
                )
                prev_file = f"{args.weights_dir}/checkpoint_epoch{current_epoch-1}_iter{current_iteration}.pt"
                if os.path.isfile(prev_file):
                    os.remove(prev_file)
            
        # Validation Epoch
        if (current_epoch + 1) % args.validate_every_epoch == 0 or current_epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            for i, vbatch in tqdm(enumerate(val_loader),total=len(val_loader)):
                noisy_image_v, actual_noise_v, time_v, label_v = vbatch
                
                noisy_image_v = noisy_image_v.to(args.device)
                actual_noise_v = actual_noise_v.to(args.device)
                time_v = time_v.to(args.device)
                label_v = label_v.to(args.device)

                with torch.no_grad():
                    predicted_noise_v = model(noisy_image_v, time_v)
                    l_v = criterion(predicted_noise_v, actual_noise_v)   
                val_loss += l_v.item()
            val_loss /= (i + 1)
            print(f"Validation Loss: {val_loss}")
            logging.info(f"Validation Loss {val_loss} at epoch {current_epoch}, iter {current_iteration}")
            
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    "best_model.pt",
                    model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                    current_epoch, current_iteration, val_loss,
                    args, args.weights_dir
                )
                print(f"Saving model as BEST MODEL")
            model.train()

        # Epoch-end save
        if (current_epoch + 1) % args.save_every_n_epochs == 0 or current_epoch == epochs -1 :
            avg_loss = epoch_loss / (total_batches - skip_iteration if current_epoch == start_epoch else total_batches)
            save_checkpoint(
                f"epoch{current_epoch+1}_end.pt",
                model.state_dict(), optimizer.state_dict(), scheduler.state_dict(),
                current_epoch, current_iteration, avg_loss,
                args, args.weights_dir
            )
            
        # after first resume epoch, reset skip_iteration
        skip_iteration = -1      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file_path", type=str, default="mnist_data.pkl")
    parser.add_argument("--val_file_path", type=str, default="mnist_data.pkl")

    parser.add_argument("--steps", type=int, default=1000)

    parser.add_argument("--img_ch", type=int, default=1)
    parser.add_argument("--time_dim", type=int, default=128)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--save_every_n_epochs", type=int, default=75)
    parser.add_argument("--save_every_n_iteration_percentage", type=int, default=110)
    parser.add_argument("--validation_every_n_iteration_percentage", type=int, default=110)
    parser.add_argument("--validate_every_epoch", type=int, default=25)    
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--weights_dir", type=str, default="weights_5")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Enter the path to resume training
    weights_path = None
    args.resume_from = weights_path

    # While resuming, all the args are overwritten by the args in the pt file 
    checkpoint_config = None
    resume = False
    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint_config = torch.load(args.resume_from)
        args = Namespace(**checkpoint_config["training_config"])
        args.resume_from = weights_path
        resume = True
    
    train(args, checkpoint_config, resume)