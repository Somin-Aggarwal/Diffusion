import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataloader import linear_schedule, cosine_schedule
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = "weights_5/best_model.pt"
weights = torch.load(weights_path, map_location=device)
training_config = weights['training_config']
from model import UNET  # make sure model.py is in path
model = UNET(image_channels=training_config['img_ch'],
             time_dim=training_config['time_dim']).to(device)
model.load_state_dict(weights["model_state_dict"])
model.eval()

batch_size = 6
steps = training_config['steps']
image_size = 32
try:
    schedule = training_config['schedule']
    if schedule == "linear":
        data = linear_schedule(steps)
        beta_t = data['betas'].to(device)
        alpha_t = data['alphas']
        alpha_t_dash = data['alphas_cumprod']
    if schedule == "cosine":
        data = cosine_schedule(steps)
        beta_t = data['betas'].to(device)
        alpha_t = data['alphas']
        alpha_t_dash = data['alphas_cumprod']
except KeyError:
    schedule = "linear"
    data = linear_schedule(steps)
    beta_t = data['betas'].to(device)
    alpha_t = data['alphas']
    alpha_t_dash = data['alphas_cumprod']


print(f"Steps : {steps} | Schedule : {schedule}")

new_image = torch.randn(size=(batch_size, training_config['img_ch'], image_size, image_size), device=device)
images = [[] for _ in range(batch_size)]  # each sublist holds frames for one sample

M = 1   

sqrt_one_minus_alphas_cumprod = data['sqrt_one_minus_alphas_cumprod']

with torch.no_grad():
    for time_step in tqdm(reversed(range(1, steps + 1))):
        for m in range(M):
            t_idx = time_step - 1
            t = torch.tensor(time_step, dtype=torch.float32, device=device).view(1, 1, 1, 1)

            predicted_noise = model(new_image, t)
            pred_score = - ( predicted_noise / sqrt_one_minus_alphas_cumprod[t_idx] )

            # eps = 10e-5 * (sqrt_one_minus_alphas_cumprod[t_idx]**2 / sqrt_one_minus_alphas_cumprod[0]**2)
            
            alpha_bar_curr = alpha_t_dash[t_idx]
            eps = 1e-5 * (1-alpha_t_dash[t_idx])/(1-alpha_t_dash[0])
            new_image = new_image + eps * pred_score + torch.sqrt(2*eps)*torch.randn_like(new_image)

        # Normalize batch for visualization
        min_val = new_image.amin(dim=(1, 2, 3), keepdim=True)
        max_val = new_image.amax(dim=(1, 2, 3), keepdim=True)
        norm = (new_image - min_val) / (max_val - min_val + 1e-8)

        # Convert each sample to numpy and store; ensure each stored image is either (H,W) or (H,W,3)
        for i in range(batch_size):
            sample = norm[i].detach().cpu().numpy()  # (C, H, W)
            c, h, w = sample.shape

            if c == 1:
                # grayscale -> convert to HxW then duplicate to RGB for display if desired
                gray = sample.squeeze(0)  # (H, W)
                # If you want to keep single-channel display, store gray (H,W)
                # But user requested convert single channel to 3 channels — do that now:
                rgb = np.repeat(gray[:, :, None], 3, axis=2)  # (H, W, 3)
                images[i].append(rgb.astype(np.float32))
            elif c == 3:
                # (C,H,W) -> (H,W,3)
                rgb = np.transpose(sample, (1, 2, 0))
                images[i].append(rgb.astype(np.float32))
            else:
                # Other channel counts: make a best-effort conversion
                if c > 3:
                    # take first 3 channels
                    rgb = np.transpose(sample[:3], (1, 2, 0))
                    images[i].append(rgb.astype(np.float32))
                else:
                    # c == 2 (or other <3): pad/duplicate channels to make 3
                    # build channels list
                    chans = [sample[j] for j in range(c)]
                    while len(chans) < 3:
                        chans.append(chans[-1])  # duplicate last channel
                    stacked = np.stack(chans[:3], axis=-1)  # (H, W, 3)
                    images[i].append(stacked.astype(np.float32))

print(len(images[0]))

fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
if batch_size == 1:
    axes = [axes]

ims = []
for i, ax in enumerate(axes):
    ax.axis("off")
    first_img = images[i][0]
    # choose grayscale colormap only for single-channel arrays
    if first_img.ndim == 2:
        im = ax.imshow(first_img, cmap="gray", vmin=0, vmax=1)
    else:
        # RGB image (H,W,3)
        im = ax.imshow(first_img, vmin=0, vmax=1)
    ims.append(im)

def update_frame(frame_idx):
    for i in range(batch_size):
        frame = images[i][frame_idx]
        ims[i].set_data(frame)
    return ims

ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=len(images[0]),
    interval=2,  # increased to 50ms for visible frames; change as desired
    blit=True,
)

plt.tight_layout()
plt.show()

final_images = [images[i][-1] for i in range(batch_size)]

fig_final, axes_final = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
if batch_size == 1:
    axes_final = [axes_final]

for i, ax in enumerate(axes_final):
    img = final_images[i]
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img, vmin=0, vmax=1)
    ax.set_title(f"Sample {i+1}")
    ax.axis("off")

plt.suptitle("Final Generated Images (x₀)", fontsize=14)
plt.tight_layout()
plt.show()