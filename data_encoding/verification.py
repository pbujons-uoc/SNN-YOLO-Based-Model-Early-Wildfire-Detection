"""
Visualization script for encoded spike datasets.

This script:
1. Loads encoded .pt files (latency or poisson) from datasets/{encoding}_4/
2. Reconstructs images from spike tensors
3. Draws bounding boxes from YOLO labels
4. Creates side-by-side visualizations: original image vs encoded reconstruction
5. Generates 3 examples: fire-only, smoke-only, and fire+smoke

Usage:
    python data_encoding/verification.py --encoding latency --time-steps 4
    python data_encoding/verification.py --encoding poisson --time-steps 4
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def decode_spikes_to_image(spikes_tensor, encoding_type='poisson'):
    """
    Reconstructs an RGB image from a spike tensor [T, C, H, W].
    
    For rate/poisson encoding:
      - Average spike count over time steps (more spikes = brighter)
      - Simply: intensity = mean(spikes along time axis)
    
    For latency encoding (TTFS): 
      - Find the time at which each pixel first spikes
      - Map fire time back to intensity: intensity = 1 - (fire_time / (T-1))
    
    Args:
        spikes_tensor: [T, C, H, W] with binary values 0 or 1
        encoding_type: 'poisson', 'rate', or 'latency'
    
    Returns: numpy array (H, W, C) uint8 0-255
    """
    T, C, H, W = spikes_tensor.shape
    spikes_tensor = spikes_tensor.float()
    
    if encoding_type in ['poisson', 'rate']:
        # Rate/Poisson: average spikes over time dimension
        # [T, C, H, W] -> [C, H, W] by averaging along T
        img_float = spikes_tensor.mean(dim=0)  # [C, H, W], values 0-1
        
        # Transpose to [H, W, C] for numpy
        img_float = img_float.permute(1, 2, 0)
        
        # Convert to numpy and scale to 0-255
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # Convert to grayscale to eliminate artificial color artifacts
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return img_np
    
    elif encoding_type == 'latency':
        # Latency (TTFS): find first spike time for each pixel
        # Create tensor of first spike times: [C, H, W]
        # If pixel never spikes, set to T (maximum latency)
        
        # Find first spike along time dimension
        # argmax returns index of first 1, or 0 if all zeros
        first_spike_times = torch.argmax(spikes_tensor.float(), dim=0)  # [C, H, W]
        
        # For pixels that never spike (all zeros), mark them differently
        any_spike = torch.any(spikes_tensor, dim=0).float()  # [C, H, W]
        
        # Reconstruct intensity from spike time
        # Early spike (t≈0) → high intensity, Late spike (t≈T-1) → low intensity
        # intensity = 1 - (spike_time / (T-1))
        if T > 1:
            img_float = 1.0 - (first_spike_times.float() / (T - 1))
        else:
            img_float = torch.ones_like(first_spike_times).float()
        
        # Mask out pixels that never spiked (set to 0)
        img_float = img_float * any_spike
        
        # Transpose to [H, W, C] for numpy
        img_float = img_float.permute(1, 2, 0)
        
        # Convert to numpy and scale to 0-255
        img_np = img_float.cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        
        # Convert to grayscale to eliminate artificial color artifacts
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        return img_np
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def parse_yolo_label(label_path, img_h, img_w):
    """
    Parses YOLO format label file.
    Returns: list of (cls, x1, y1, x2, y2) in pixel coordinates.
    """
    if not os.path.exists(label_path):
        return []
    
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            # YOLO format: cls x_center y_center width height (normalized 0-1)
            x_c, y_c, w, h = map(float, parts[1:5])
            
            # Convert to pixel coordinates
            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h
            
            x1 = int(x_c - w/2)
            y1 = int(y_c - h/2)
            x2 = int(x_c + w/2)
            y2 = int(y_c + h/2)
            
            boxes.append((cls, x1, y1, x2, y2))
    
    return boxes


def draw_boxes_on_image(img, boxes, class_names={0: 'smoke', 1: 'fire'}):
    """
    Draws bounding boxes on an image.
    boxes: list of (cls, x1, y1, x2, y2)
    Returns: annotated image
    """
    img_copy = img.copy()
    
    for cls, x1, y1, x2, y2 in boxes:
        color = (0, 0, 255) if cls == 1 else (255, 0, 0)  # fire=red, smoke=blue
        label = class_names.get(cls, str(cls))
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img_copy, label, (x1, y1 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


def create_side_by_side(img_orig, img_encoded, title_left="Original", title_right="Encoded"):
    """
    Creates a side-by-side comparison image.
    """
    h1, w1 = img_orig.shape[:2]
    h2, w2 = img_encoded.shape[:2]
    
    # Resize to same height if needed
    target_h = max(h1, h2)
    if h1 != target_h:
        img_orig = cv2.resize(img_orig, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        img_encoded = cv2.resize(img_encoded, (int(w2 * target_h / h2), target_h))
    
    # Concatenate horizontally
    combined = np.hstack([img_orig, img_encoded])
    
    # Add titles
    cv2.putText(combined, title_left, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, title_right, (img_orig.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return combined


def find_examples(encoded_dir, label_dir):
    """
    Finds examples of: fire-only, smoke-only, fire+smoke.
    Returns: dict with keys 'fire', 'smoke', 'both' → pt file path
    """
    examples = {'fire': None, 'smoke': None, 'both': None}
    
    pt_files = list(Path(encoded_dir).glob('*.pt'))
    
    for pt_file in pt_files:
        # Get corresponding label
        label_file = Path(label_dir) / (pt_file.stem + '.txt')
        if not label_file.exists():
            continue
        
        # Read classes
        classes = set()
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.add(int(parts[0]))
        
        if not classes:
            continue
        
        # Classify
        if classes == {1} and examples['fire'] is None:
            examples['fire'] = pt_file
        elif classes == {0} and examples['smoke'] is None:
            examples['smoke'] = pt_file
        elif classes == {0, 1} and examples['both'] is None:
            examples['both'] = pt_file
        
        # Stop if all found
        if all(v is not None for v in examples.values()):
            break
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Verify encoded spike datasets with visualization")
    parser.add_argument('--encoding', type=str, required=True, choices=['latency', 'poisson'],
                        help='Encoding type (latency or poisson)')
    parser.add_argument('--time-steps', type=int, default=4, help='Number of time steps')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Output directory (default: data_encoding/{encoding})')
    
    args = parser.parse_args()
    
    # Paths
    encoding_name = f"{args.encoding}_{args.time_steps}"
    encoded_dataset = f"datasets/{encoding_name}/train/images"
    label_dir = f"datasets/{encoding_name}/train/labels"
    original_dataset = "datasets/D-Fire/train/images"
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"data_encoding/{args.encoding}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Encoding: {args.encoding}")
    print(f"Encoded dataset: {encoded_dataset}")
    print(f"Original dataset: {original_dataset}")
    print(f"Output directory: {output_dir}")
    
    # Find examples
    print("\nSearching for examples...")
    examples = find_examples(encoded_dataset, label_dir)
    
    for category, pt_file in examples.items():
        if pt_file is None:
            print(f"  [WARNING] No example found for category: {category}")
        else:
            print(f"  {category}: {pt_file.name}")
    
    # Process each example
    for category, pt_file in examples.items():
        if pt_file is None:
            continue
        
        print(f"\nProcessing {category} example: {pt_file.name}")
        
        # Load encoded spikes
        spikes = torch.load(pt_file)  # [T, C, H, W]
        print(f"  Spikes shape: {spikes.shape}")
        
        # Reconstruct image with encoding-aware decoding
        img_reconstructed = decode_spikes_to_image(spikes, encoding_type=args.encoding)
        h, w = img_reconstructed.shape[:2]
        
        # Load original image
        orig_img_path = Path(original_dataset) / (pt_file.stem + '.jpg')
        if not orig_img_path.exists():
            orig_img_path = Path(original_dataset) / (pt_file.stem + '.png')
        
        if not orig_img_path.exists():
            print(f"  [WARNING] Original image not found: {orig_img_path}")
            continue
        
        img_original = cv2.imread(str(orig_img_path))
        if img_original is None:
            print(f"  [ERROR] Could not load original image")
            continue
        
        # Parse labels
        label_path = Path(label_dir) / (pt_file.stem + '.txt')
        boxes = parse_yolo_label(str(label_path), h, w)
        boxes_orig = parse_yolo_label(str(label_path), img_original.shape[0], img_original.shape[1])
        
        print(f"  Found {len(boxes)} bounding boxes")
        
        # Draw boxes
        img_orig_annotated = draw_boxes_on_image(img_original, boxes_orig)
        img_recon_annotated = draw_boxes_on_image(img_reconstructed, boxes)
        
        # Create side-by-side
        combined = create_side_by_side(img_orig_annotated, img_recon_annotated, 
                                      title_left="Original + Labels",
                                      title_right=f"{args.encoding.capitalize()} Encoded + Labels")
        
        # Save
        output_path = os.path.join(output_dir, f"{category}_comparison.jpg")
        cv2.imwrite(output_path, combined)
        print(f"  Saved: {output_path}")
    
    print(f"\n✅ Done! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
