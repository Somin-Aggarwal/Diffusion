import pickle 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }

def cosine_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)

    alphas_cumprod = torch.cos((((x / T) + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so ᾱ₀ = 1

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-20, max=0.9999)

    alphas = 1.0 - betas

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod[1:],  # drop ᾱ₀ to align indexing
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod[1:]),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod[1:]),
    }

class DiffusionDataset(Dataset):
    def __init__(self, file_path:str, mode:str, steps, schedule="linear",*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert schedule=="linear" or schedule=="cosine"
        
        self.file_path = file_path
        if mode not in ["train", "test"]:
            raise ValueError(" Mode should be wither \"train\" or \"test\"")
        self.mode = mode
        
        with open(self.file_path,"rb") as f:
            data = pickle.load(f)
      
        self.transform  = transforms.Compose([
            transforms.ToTensor(), # Image is scaled between 0 and 1
            transforms.Resize(size=(32,32), antialias=False),
        ]) # Input Range : [-1,+1]
                
        self.x = data[f"x_{self.mode}"]
        if self.x.ndim == 3:
            self.x = np.expand_dims(self.x,axis=-1)
            self.transform.transforms.append(transforms.Normalize((0.5),(0.5)))
        else:
            self.transform.transforms.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))) # ( image - 0.5 ) / 0.5
        self.y = data[f"y_{self.mode}"] 
        del data
        
        self.steps = steps
        
        if schedule == "linear":
            data = linear_schedule(steps)
        if schedule == "cosine":
            data = cosine_schedule(steps)

        self.beta_t = data["betas"]
        self.alpha_t = data["alphas"]
        
        self.sqrt_alpha_t_dash = data['sqrt_alphas_cumprod']
        self.sqrt_1_minus_alpha_t_dash = data['sqrt_one_minus_alphas_cumprod']
        
        self.val_multiple = 20
        self.bin_size = self.steps // self.val_multiple
        
    def forward_process(self, image_tensor, t):
        noise = torch.randn_like(image_tensor)
        noisy_image = self.sqrt_alpha_t_dash[t] * image_tensor + self.sqrt_1_minus_alpha_t_dash[t] * noise
        return noisy_image, noise

    def __len__(self):
        if self.mode == "test":
            return self.x.shape[0]*self.val_multiple
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == "test":
            bin_idx = idx % self.val_multiple
            bin_start = bin_idx * self.bin_size
            bin_end = bin_start + self.bin_size
            t = np.random.randint(bin_start, bin_end)
            idx =  idx // self.val_multiple 
        else:
            t = int(np.random.uniform(low=0,high=self.steps))
            
        noisy_image, noise = self.forward_process( self.transform(self.x[idx]) ,t )
        return noisy_image, noise, t, self.y[idx]


class CelebFaceDataset(Dataset):
    def __init__(self, folder_path: str, steps: int, schedule="linear"):
        super().__init__()
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Directory {folder_path} not found.")
        
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {folder_path}")

        assert schedule in ["linear", "cosine"], "Schedule must be 'linear' or 'cosine'"
        self.steps = steps
        
        if schedule == "linear":
            data = linear_schedule(steps)
        else:
            data = cosine_schedule(steps)

        self.sqrt_alpha_t_dash = data['sqrt_alphas_cumprod']
        self.sqrt_1_minus_alpha_t_dash = data['sqrt_one_minus_alphas_cumprod']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Scales [0, 1] to [-1, 1]
        ])

    def forward_process(self, image_tensor, t):
        noise = torch.randn_like(image_tensor)
        # Ensure indexing works if these are arrays
        sqrt_alpha = self.sqrt_alpha_t_dash[t]
        sqrt_one_minus_alpha = self.sqrt_1_minus_alpha_t_dash[t]
        
        noisy_image = sqrt_alpha * image_tensor + sqrt_one_minus_alpha * noise
        return noisy_image, noise

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        
        t = np.random.randint(0, self.steps)
        
        noisy_image, noise = self.forward_process(image_tensor, t)
        
        # Returning (noisy_img, noise, timestep)
        # Note: 'y' (labels) is omitted here unless your filenames contain labels
        return noisy_image, noise, t

def forward_process():
    
    steps = [i for i in range(1,1001)]
    linear_data = linear_schedule(T=1000)
    cosine_data = cosine_schedule(T=1000)
    
    with open("../cifar10_data.pkl","rb") as file:
        data = pickle.load(file)
    test_images = data['x_test']
    
    image = test_images[np.random.randint(low=0,high=test_images.shape[0])]

    dataset = DiffusionDataset(file_path="../cifar10_data.pkl",mode="test",steps=1000)
    image = dataset.transform(image)
    
    time_steps = [1,50,100,200,300,500,700,999]
    
    images_2 = []
    noise_2 = []
    
    current_data = linear_data
    for i in range(2):
        coeff_1 = current_data["sqrt_alphas_cumprod"]
        coeff_2 = 1 - current_data['alphas_cumprod']
        
        plt.plot(np.arange(0,1000),coeff_1, label="mean")
        plt.plot(np.arange(0,1000),coeff_2, label="variance")
        plt.plot(np.arange(0,1000),np.sqrt(coeff_2), label="std")
        plt.legend()
        plt.title("forward")
        plt.show()
        
        coeff_3 = (1 - current_data['alphas']) / (current_data['sqrt_one_minus_alphas_cumprod'] * torch.sqrt(current_data["alphas"]))
        plt.plot(np.arange(0,1000),1/torch.sqrt(current_data['alphas']), label="1/sqrt_alpha_t")
        plt.plot(np.arange(0,1000),coeff_3, label="noise_coeff")
        plt.plot(np.arange(0,1000),np.sqrt(coeff_2), label="std")
        plt.legend()
        plt.title("reverese")
        plt.show()

        
        images_list = []
        noises_list = []
        
        for time in steps:
            noise = torch.randn_like(image)
            new_image_not_norm = coeff_1[time-1] * image + coeff_2[time-1]*noise
            new_image = new_image_not_norm.permute(1,2,0).cpu().numpy()
            new_image = ( new_image - new_image.min() ) / ( new_image.max() - new_image.min())
            
            images_list.append(new_image)
            noises_list.append(noise)
        
        images_2.append(images_list)
        noise_2.append(noises_list)
        
        current_data = cosine_data
    
    fig, axes = plt.subplots(nrows=2,ncols=len(time_steps))
    
    for i,time in enumerate(time_steps):
        
        axes[0, i].imshow(images_2[0][time-1]) 
        axes[0, i].set_title(f'Time: {time_steps[i]}') 
        axes[0, i].axis('off')

        axes[1, i].imshow(images_2[1][time-1])
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()    

def get_stats(data_name, schedule, steps, target_samples):
    """
    Runs the dataloader and calculates mean/std trajectories 
    for a specific dataset and schedule configuration.
    """
    print(f"\n--- Processing: {data_name.upper()} | Schedule: {schedule} ---")
    
    # Dynamic path handling
    file_path = f"../{data_name}_data.pkl"
    
    dataset = DiffusionDataset(file_path=file_path, mode="test", steps=steps, schedule=schedule)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    step_means = torch.zeros(steps)
    step_stds = torch.zeros(steps)
    step_counts = torch.zeros(steps)

    total_samples_processed = 0
    
    # Create a progress bar
    pbar = tqdm(total=target_samples, desc=f"{data_name}-{schedule}")

    while total_samples_processed < target_samples:
        for batch in loader:
            noisy_imgs, _, t, _ = batch
            
            # Identify current batch size (handling last batch drop-off)
            curr_b_size = noisy_imgs.shape[0]

            # Flatten [B, C, H, W] -> [B, Pixels]
            flat_imgs = noisy_imgs.view(curr_b_size, -1)
            
            batch_means = flat_imgs.mean(dim=1) # [B]
            batch_stds = flat_imgs.std(dim=1)   # [B]
            
            t = t.long().cpu()
            
            # Accumulate
            step_means.index_add_(0, t, batch_means)
            step_stds.index_add_(0, t, batch_stds)
            
            ones = torch.ones_like(t, dtype=torch.float)
            step_counts.index_add_(0, t, ones)

            processed_now = len(t)
            total_samples_processed += processed_now
            pbar.update(processed_now)

            if total_samples_processed >= target_samples:
                break
    
    pbar.close()

    # Avoid div by zero
    step_counts = step_counts.clamp(min=1)
    
    avg_means = (step_means / step_counts).numpy()
    avg_stds = (step_stds / step_counts).numpy()
    
    return avg_means, avg_stds

def main():
    # Global Settings
    steps = 1000
    target_samples = 200000  # Adjusted to 50k as per instructions
    
    # Configurations to test
    datasets = ["mnist", "cifar10"]
    schedules = ["linear", "cosine"]
    
    # Setup Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    time_steps = range(steps)
    
    # Loop through all 4 combinations
    for data_name in datasets:
        for schedule in schedules:
            
            # Get stats for this specific combo
            avg_means, avg_stds = get_stats(data_name, schedule, steps, target_samples)
            
            # Create a label for the legend
            label_str = f"{data_name.upper()} - {schedule}"
            
            # Plot on shared axes
            # Axis 0: Means
            axes[0].plot(time_steps, avg_means, label=label_str, alpha=0.8)
            
            # Axis 1: Stds
            axes[1].plot(time_steps, avg_stds, label=label_str, alpha=0.8)

    # --- Final Plot Styling ---
    
    # Mean Plot Settings
    axes[0].set_title(f'Mean Pixel Value Trajectories ({target_samples} samples)')
    axes[0].set_xlabel('Time Step (t)')
    axes[0].set_ylabel('Mean Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Std Plot Settings
    axes[1].set_title(f'Std Dev Trajectories ({target_samples} samples)')
    axes[1].set_xlabel('Time Step (t)')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("diffusion_schedule_comparison.jpg")
    plt.show()
    print("\nComparison graph saved and displayed.")

if __name__ == "__main__":
    
    train_dataset = CelebFaceDataset(folder_path="../archive/training",
                                     steps=1000,
                                     schedule="cosine")
    
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,num_workers=12)
    
    for i,batch in tqdm(enumerate(train_dataloader)):
        pass
    
    
    