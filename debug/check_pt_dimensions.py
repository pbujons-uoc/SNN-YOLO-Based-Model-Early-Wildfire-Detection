# check_shape.py
import torch
from pathlib import Path

# Path to the .pt file
pt_path = Path("./AoF00000.pt")

# Load the tensor / object
data = torch.load(pt_path, map_location="cpu")

# Print type and shape (if applicable)
print("Type:", type(data))

if hasattr(data, "shape"):
    print("Shape:", data.shape)
elif isinstance(data, dict):
    print("Dictionary keys:", data.keys())
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape {v.shape}")
        else:
            print(f"  {k}: type {type(v)}")
else:
    print("Loaded object has no .shape attribute")
