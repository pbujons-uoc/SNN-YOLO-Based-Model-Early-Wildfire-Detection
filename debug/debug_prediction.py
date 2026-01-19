
import os
import sys

# -------------------------------------------------------------------------
# CRITICAL PATH SETUP
# We must ensure 'ultralytics' is imported from 'SpikeYOLO_Encoded/ultralytics'
# NOT from the system site-packages.
# -------------------------------------------------------------------------

# 1. Get absolute path to the wrapper folder
current_dir = os.getcwd() # e.g. /home/pau/SECURE_DATA
wrapper_dir = os.path.join(current_dir, "SpikeYOLO_Encoded")

# 2. Force it to the FRONT of sys.path
if wrapper_dir not in sys.path:
    sys.path.insert(0, wrapper_dir)

print(f"DEBUG: sys.path[0] is {sys.path[0]}")

# 3. NOW import ultralytics
import ultralytics
print(f"DEBUG: ultralytics imported from: {ultralytics.__file__}")

# 4. Import YOLO
from ultralytics import YOLO
import torch

# -------------------------------------------------------------------------
# DEBUG LOGIC
# -------------------------------------------------------------------------

# Config
model_path = "Results/SpikeYOLO_Encoded/latency_4_extended/weights/last.pt"
image_path = "datasets/latency_4/val/images" 

def debug_inference():
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Find a .pt file
    if not os.path.exists(image_path):
         print(f"Image path not found: {image_path}")
         return
         
    pt_files = [f for f in os.listdir(image_path) if f.endswith('.pt')]
    if not pt_files:
        print(f"No .pt files found in {image_path}")
        return
    
    # NEW: Find a file that actually has labels to test True Positive performance
    label_path_dir = image_path.replace("images", "labels")
    sample_file = None
    
    print(f"Scanning {len(pt_files)} files for a positive sample...")
    for f in pt_files:
        lbl_name = f.replace('.pt', '.txt')
        lbl_full = os.path.join(label_path_dir, lbl_name)
        if os.path.exists(lbl_full) and os.path.getsize(lbl_full) > 0:
            sample_file = os.path.join(image_path, f)
            print(f"Found labeled sample: {f}")
            break
            
    if sample_file is None:
        print("Warning: No labeled images found. Defaulting to first file.")
        sample_file = os.path.join(image_path, pt_files[0])
    
    print(f"Testing on {sample_file}...")
    
    # Load data
    img = torch.load(sample_file) # [T, C, H, W]
    print(f"Input shape: {img.shape}")
    
    # Preprocess
    if img.dim() == 4:
        img = img.unsqueeze(0) # [1, T, C, H, W]
    
    img = img.float()

    # CRITICAL: Resize to 640x640 (or multiple of 32)
    # The model fails if H is not divisible by 32 (720 / 32 = 22.5)
    # We use nearest interpolation to keep spikes binary-ish
    # img shape is [B, T, C, H, W]. 
    # interpolate expects [Batch, Channels, H, W]. We need to flatten T/C or loop.
    
    B, T, C, H, W = img.shape
    img_reshaped = img.view(B * T, C, H, W)
    img_resized = torch.nn.functional.interpolate(img_reshaped, size=(640, 640), mode='nearest')
    img = img_resized.view(B, T, C, 640, 640)
    
    print(f"Resized input shape: {img.shape}")
    
    # Run Inference
    # Use internal model directly to bypass preprocessing checks if needed, 
    # but YOLO() call usually handles it if we pass a standard image. 
    # Passing a TENSOR to YOLO(...) predict/call is tricky.
    # Safe bet: use model.model(x) directly.
    
    device = next(model.model.parameters()).device
    img = img.to(device)
    model.model.eval()
    
    print("Running forward pass...")
    with torch.no_grad():
        preds = model.model(img)
        
        # Output parsing
        if isinstance(preds, tuple):
             # Usually (pred, proto) or (pred, hidden)
             output = preds[0]
        else:
             output = preds

        
        # Analyze Confidence
        # YOLOv8 Output is [B, 4+C, Anchors]
        # Box: 0-4
        # Cls: 4-end (Logits? Or Probs?)
        # Let's check ranges.
        
        box_data = output[:, :4, :]
        cls_data = output[:, 4:, :]
        
        print(f"\nOutput Shape: {output.shape}")
        
        print("\n--- Box Coordinates (xywh) Statistics ---")
        print(f"Min: {box_data.min().item():.4f}")
        print(f"Max: {box_data.max().item():.4f}")
        print(f"Mean: {box_data.mean().item():.4f}")

        print("\n--- Class Scores Statistics ---")
        print(f"Min: {cls_data.min().item():.6f}")
        print(f"Max: {cls_data.max().item():.6f}")
        print(f"Mean: {cls_data.mean().item():.6f}")
        
        # Check if they are logits or probs
        if cls_data.max() > 1.0 or cls_data.min() < 0.0:
            print("NOTE: Scores appear to be Logits (Raw). They need Sigmoid.")
            cls_probs = torch.sigmoid(cls_data)
        else:
            print("NOTE: Scores appear to be Probabilities (0-1).")
            cls_probs = cls_data
            
        print(f"\n--- Probabilities (Sigmoid) ---")
        print(f"Max Prob: {cls_probs.max().item():.6f}")
        
        threshold = 0.001
        high_conf_count = (cls_probs > threshold).sum().item()
        print(f"Anchors > {threshold}: {high_conf_count}")
        
        threshold_mid = 0.01
        mid_conf_count = (cls_probs > threshold_mid).sum().item()
        print(f"Anchors > {threshold_mid}: {mid_conf_count}")

        threshold_high = 0.1
        higher_conf_count = (cls_probs > threshold_high).sum().item()
        print(f"Anchors > {threshold_high}: {higher_conf_count}")

if __name__ == "__main__":
    debug_inference()
