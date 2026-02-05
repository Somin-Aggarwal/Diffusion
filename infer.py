import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from my_model import UNet, UNet_conditional, DiffusionClassifier
from sampling import SamplingClass

device = "cuda" if torch.cuda.is_available() else "cpu"

# weights_path = "mnist_clsfree/epoch400.pt"
weights_path = "cifar_clsfree/epoch800.pt"

weights = torch.load(weights_path, map_location=device)

config = weights['training_config']
print(config)
mcfg = config['model']

# ---- Initialize Model ----
model = UNet_conditional(
    img_ch=mcfg['img_ch'],
    base_ch=mcfg['base_ch'],
    ch_mul=mcfg['ch_mul'],
    attn=mcfg['attn'],
    n_resblocks=mcfg['n_resblocks'],
    steps=config['steps'],
    tdim=mcfg['tdim'],
    n_classes=config['n_classes']
).to(device)
model.load_state_dict(weights["model_state_dict"])
model.eval()

# ---- Setup Hyperparameters ----
batch_size = 10
steps = config['steps']
image_size = 32
schedule = config.get('schedule', 'cosine')

SamplingObject = SamplingClass(
    model=model,
    batch_size=batch_size,
    schedule=schedule,
    steps=steps,
    img_ch=mcfg["img_ch"],
    n_classes=config["n_classes"],
    device=device
)

null_idx = config["n_classes"]
null_labels = [null_idx for _ in range(batch_size)]
labels = [i for i in range(10)]
# labels = [0,0,1,1,2,2,3,3,4,4]
weight = 0.9
N_corrector = 1
time_steps = [i for i in range(1,1001,10)]
time_steps.append(1000)

x_ts = SamplingObject.DDIM_Interpolation(time_steps)
SamplingObject.visualize(x_ts)
SamplingObject.visualize_stats(x_ts, sampling_strategy="DDIM")

# x_ts = SamplingObject.Ancestral2(labels,weight)
# SamplingObject.visualize(x_ts)
# SamplingObject.visualize_stats(x_ts, sampling_strategy="Ancestral_sampling")

# x_ts = SamplingObject.PredictorCorrector(labels,N_corrector)
# SamplingObject.visualize(x_ts)
# SamplingObject.visualize_stats(x_ts, sampling_strategy=f"PC Sampling ({N_corrector} Corrector Steps)")

# x_ts = SamplingObject.LangevinDynamics2(labels,weight)
# SamplingObject.visualize(x_ts)
# SamplingObject.visualize_stats(x_ts, sampling_strategy="Langevin Dynamics")

# x_ts = SamplingObject.ClassifierFree(labels,weight)
# SamplingObject.visualize(x_ts)
# SamplingObject.visualize_stats(x_ts, sampling_strategy="Classifier Free Guidance")

# x_ts = SamplingObject.ConditionalClassifierGuided(classifier_model,labels,weight)
# SamplingObject.visualize(x_ts)
# SamplingObject.visualize_stats(x_ts, sampling_strategy="Classifier Free Guidance")



