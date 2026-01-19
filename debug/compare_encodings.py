
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. Internal Implementation "Latency" (Linear) ---
def internal_latency_encode(img_tensor, T=4):
    # img_tensor: [C, H, W] float 0..1
    # Logic from MS_GetT in yolo_spikformer.py
    
    # "fire_times: 0 (brightest) to T-1 (darkest)"
    fire_times = ((1.0 - img_tensor) * (T - 1)).long()
    
    T_idx = torch.arange(T).view(T, 1, 1, 1)
    # Broadcasting [1, C, H, W]
    spikes = (T_idx == fire_times.unsqueeze(0)).float()
    return spikes # [T, C, H, W]

# --- 2. External Implementation "Rank Order" (Sorting) ---
def external_rank_order_encode(img_numpy, T=4):
    # img_numpy: [H, W, C] uint8 0..255
    # Logic from rank_order_encoding.py
    
    norm = img_numpy.astype(np.float32) / 255.0
    norm = norm.transpose(2, 0, 1)
    tensor = torch.from_numpy(norm) # [C, H, W]
    
    C, H, W = tensor.shape
    spikes = torch.zeros((T, C, H, W), dtype=torch.float32)
    
    for c in range(C):
        channel_data = tensor[c]
        flat_c = channel_data.view(-1)
        num_pixels = flat_c.numel()
        
        # Argsort: dark -> bright
        sorted_indices = torch.argsort(flat_c)
        ranks = torch.zeros_like(flat_c, dtype=torch.long)
        ranks[sorted_indices] = torch.arange(num_pixels)
        
        norm_ranks = ranks.float() / num_pixels
        
        # bright (high rank) -> time 0
        times = ((1.0 - norm_ranks) * T * 0.99999).long()
        
        for t in range(T):
            mask = (times == t).view(H, W)
            spikes[t, c] = mask.float()
            
    return spikes

# --- Compare ---
def compare_encodings():
    # Create a gradient image to clearly see the difference
    h, w = 100, 256
    gradient = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img_bgr = cv2.merge([gradient, gradient, gradient])
    
    # 1. Internal Latency
    img_tensor = torch.from_numpy(img_bgr.transpose(2, 0, 1)).float() / 255.0
    spikes_internal = internal_latency_encode(img_tensor, T=4)
    
    # 2. External Rank Order
    spikes_external = external_rank_order_encode(img_bgr, T=4)
    
    # Plotting
    fig, axes = plt.subplots(3, 4, figsize=(15, 8))
    
    # Row 0: Original
    axes[0, 0].imshow(img_bgr, cmap='gray')
    axes[0, 0].set_title("Original Gradient")
    for j in range(1, 4): axes[0, j].axis('off')
    
    # Row 1: Internal (Linear Latency)
    for t in range(4):
        axes[1, t].imshow(spikes_internal[t, 0].numpy(), cmap='binary', vmin=0, vmax=1)
        axes[1, t].set_title(f"Internal T={t}")
        axes[1, t].axis('off')
        
    # Row 2: External (Rank Order)
    for t in range(4):
        axes[2, t].imshow(spikes_external[t, 0].numpy(), cmap='binary', vmin=0, vmax=1)
        axes[2, t].set_title(f"External (Rank) T={t}")
        axes[2, t].axis('off')
        
    plt.tight_layout()
    plt.savefig("encoding_comparison.png")
    print("Comparison saved to encoding_comparison.png")
    
    # Calculate difference
    diff = torch.abs(spikes_internal - spikes_external).sum()
    print(f"Total pixel difference count: {diff.item()}")

if __name__ == "__main__":
    compare_encodings()
