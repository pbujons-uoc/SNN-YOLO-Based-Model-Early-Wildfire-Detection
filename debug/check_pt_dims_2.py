
import torch
import sys

try:
    pt_data = torch.load("AoF00000.pt")
    print(f"Type: {type(pt_data)}")
    if hasattr(pt_data, 'shape'):
        print(f"Shape: {pt_data.shape}")
    else:
        print("Data is not a tensor.")
except Exception as e:
    print(f"Error loading file: {e}")
