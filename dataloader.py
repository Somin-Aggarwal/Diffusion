import pickle 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

'''
I will normalize the image from [-1.+1] intially at t =0
I will using the formula os the forward process on the image 
which will change the scaaling but I should not rescale it again
as the resultant should have a different mean and variance which is the 
point of the forward process so that at time = 1 my image is convert to 
pure noise, rescaling it will destroy the information
'''

'''
PRoblem with val is that if I use t as random in val then my val loss will
never be consistent, but I cannot make all the pairs as the datasize would 
fo to number_of_test_samples * np_of_steps
'''

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

    # compute f(t)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize so ᾱ₀ = 1

    # derive betas
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=0.0001, max=0.9999)

    alphas = 1.0 - betas

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod[1:],  # drop ᾱ₀ to align indexing
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod[1:]),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod[1:]),
    }

class MNISTdatasetDiffusion(Dataset):
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
        ])
        
        self.x = np.expand_dims(data[f"x_{self.mode}"],axis=-1)
        self.y = data[f"y_{self.mode}"]        
        del data
        
        self.steps = steps
        
        if schedule == "linear":
            data = linear_schedule(steps)
        if schedule == "cosine":
            data = cosine_schedule(steps)

        self.beta_t = data["betas"]
        self.alpha_t = data["alphas"]
        
        self.alpha_t_dash = data["alphas_cumprod"]
        
        self.sqrt_alpha_t_dash = torch.sqrt(self.alpha_t_dash)
        self.sqrt_1_minus_alpha_t_dash = torch.sqrt(1 - self.alpha_t_dash)
        
        self.val_multiple = 20
        self.bin_size = self.steps // self.val_multiple
        
    def convert(self, image_tensor, t):
        # image tensor ranges from [0,1] because of ToTensor() transformation
        image_tensor = 2*image_tensor - 1 # Scalling image to [-1,+1]
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
            
        noisy_image, noise = self.convert(self.transform(self.x[idx]),t)
        return noisy_image, noise, t + 1, self.y[idx]

if __name__=="__main__":
    
    
    mnist_dataset = MNISTdatasetDiffusion(file_path="mnist_data.pkl", mode="train", steps=1000)
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=8, shuffle=True, num_workers=4)
    
    print(f"Length of DataLoader : {len(mnist_dataloader)}")
    
    for i,batch in tqdm(enumerate(mnist_dataloader)):
        noisy_image, noise, time = batch
        print(noisy_image.shape, noise.shape , time)
        display_image_cifar_scaled(noisy_image, noise, time)
        # break 


    
