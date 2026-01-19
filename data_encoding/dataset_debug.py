import torch
import numpy as np

# Cargar un .pt del dataset latency_4
pt = torch.load('datasets/D-Fire/latency_4/images/train/Camera0001.pt')

print(f"Shape: {pt.shape}")
print(f"dtype: {pt.dtype}")
print(f"min: {pt.min()}, max: {pt.max()}")
print(f"unique values: {torch.unique(pt)}")
print(f"mean per timestep: {pt.float().mean(dim=(1,2,3))}")  # Average per timestep

# Check: ¿cada píxel dispara exactamente una vez?
spikes_per_pixel = pt.sum(dim=0)  # [C, H, W]
print(f"\nSpikes per pixel - min: {spikes_per_pixel.min()}, max: {spikes_per_pixel.max()}, mean: {spikes_per_pixel.float().mean()}")
print(f"Pixels that never spike: {(spikes_per_pixel == 0).sum().item()}")
print(f"Pixels that spike >1 time: {(spikes_per_pixel > 1).sum().item()}")