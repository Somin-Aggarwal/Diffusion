import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# CONFIG
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 5
steps = 1000
image_size = 32

# Mock diffusion parameters (example)
beta_t = torch.linspace(1e-4, 0.02, steps).to(device)
alpha_t = 1 - beta_t
alpha_t_dash = torch.cumprod(alpha_t, dim=0)

weights_path = "weights_5/best_model.pt"
weights = torch.load(weights_path, map_location=device)
training_config = weights['training_config']
from model import UNET, UNET_old  # make sure model.py is in path
model = UNET(image_channels=training_config['img_ch'],
             time_dim=training_config['time_dim']).to(device)
model.load_state_dict(weights["model_state_dict"])
model.eval()

# ----------------------------
# REVERSE DIFFUSION
# ----------------------------
new_image = torch.randn(size=(batch_size, 1, image_size, image_size), device=device)
images = [[] for _ in range(batch_size)]  # each sublist holds frames for one sample

#Digit to make
y = torch.tensor([6]*batch_size, device=device)  # Change digit as needed


with torch.no_grad():
    for time_step in reversed(range(1, steps + 1)):
        t_idx = time_step - 1
        t = torch.tensor(time_step, dtype=torch.float32, device=device).view(1, 1, 1, 1)

        predicted_noise = model(new_image, t)

        z = torch.randn_like(predicted_noise)
        if time_step < 100:
            z = 0
        a_t = alpha_t[t_idx]
        a_dash_t = alpha_t_dash[t_idx]
        a_dash_t_minus_one = alpha_t_dash[t_idx-1]
        b_t = beta_t[t_idx]

        new_image = (
            (1 / torch.sqrt(a_t))
            * (new_image - ((1 - a_t) / torch.sqrt(1 - a_dash_t)) * predicted_noise)
            + ((1-a_t)*(1-a_dash_t) / (1-a_dash_t_minus_one)) * z
        )

        # Normalize batch for visualization
        min_val = new_image.amin(dim=(1, 2, 3), keepdim=True)
        max_val = new_image.amax(dim=(1, 2, 3), keepdim=True)
        norm = (new_image - min_val) / (max_val - min_val + 1e-8)

        # Convert each sample to numpy and store
        for i in range(batch_size):
            img = norm[i].detach().cpu().numpy()  # (1,H,W)
            img = img.squeeze(0)  # grayscale (H,W)
            images[i].append(img)

# ----------------------------
# ANIMATION SETUP
# ----------------------------
fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
if batch_size == 1:
    axes = [axes]

ims = []
for i, ax in enumerate(axes):
    ax.axis("off")
    im = ax.imshow(images[i][0], cmap="gray", vmin=0, vmax=1)
    ims.append(im)

def update_frame(frame_idx):
    for i in range(batch_size):
        ims[i].set_data(images[i][frame_idx])
    return ims

ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=len(images[0]),
    interval=2,
    blit=True,
)

plt.tight_layout()
plt.show()


final_images = [images[i][-1] for i in range(batch_size)]

fig_final, axes_final = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
if batch_size == 1:
    axes_final = [axes_final]

for i, ax in enumerate(axes_final):
    ax.imshow(final_images[i], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Sample {i+1}")
    ax.axis("off")

plt.suptitle("Final Generated Images (x₀)", fontsize=14)
plt.tight_layout()
plt.show()
