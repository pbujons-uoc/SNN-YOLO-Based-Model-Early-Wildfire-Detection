import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def latency_encode(image, T: int):
    """
    Converts a standard RGB image into a Latency-encoded spike tensor (TTFS).
    Brighter pixels spike earlier.
    image: numpy array (H, W, C), values 0-255
    T: time steps
    Returns: Tensor [T, C, H, W] with binary values (0 or 1)
    """
    # Normalize image to 0-1
    norm = image.astype(np.float32) / 255.0
    
    # Dimensions: [H, W, C] -> [C, H, W]
    norm = norm.transpose(2, 0, 1)
    norm = torch.from_numpy(norm)
    
    # Calculate fire time: t = (1 - intensity) * (T - 1)
    # Higher intensity (1.0) -> t=0
    # Lower intensity (0.0) -> t=T-1
    fire_times = ((1.0 - norm) * (T - 1)).long()
    
    # Create spike tensor
    # Shape: [T, C, H, W]
    spikes = torch.zeros((T,) + norm.shape, dtype=torch.uint8)
    
    # Scatter spikes at correct times
    # This is efficient: for each pixel, set 1 at index 'fire_times'
    for t in range(T):
        spikes[t] = (fire_times == t).to(torch.uint8)
        
    return spikes

def process_dataset(source_dir, output_dir, T=4, img_size=None):
    """
    Recursively processes images in source_dir and saves spike tensors to output_dir.
    Mirrors the directory structure.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory {source_path} does not exist.")
        return

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in source_path.rglob('*') if p.suffix.lower() in image_extensions]
    
    print(f"Found {len(files)} images in {source_dir}")
    print(f"Encoding to Latency spikes with T={T}...")
    
    for file_path in tqdm(files):
        # Determine output path, maintaining structure
        relative_path = file_path.relative_to(source_path)
        dest_path = output_path / relative_path.with_suffix('.pt')
        
        # Skip if exists
        if dest_path.exists():
            continue

        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
             # Read Image
            img = cv2.imread(str(file_path))
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if requested
            if img_size:
                 img = cv2.resize(img, (img_size, img_size))
            
            # Encode
            spikes = latency_encode(img, T) # [T, C, H, W]
            
            # Save
            torch.save(spikes, dest_path)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Done! Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert Image Dataset to Latency Spike Tensors")
    parser.add_argument('--source', type=str, required=True, help='Path to source image directory')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--time-steps', type=int, default=4, help='Time steps (T) (default: 4)')
    parser.add_argument('--size', type=int, default=None, help='Resize images to square size (e.g. 640). Default: Keep original')
    
    args = parser.parse_args()
    
    process_dataset(args.source, args.output, T=args.time_steps, img_size=args.size)

if __name__ == "__main__":
    main()
