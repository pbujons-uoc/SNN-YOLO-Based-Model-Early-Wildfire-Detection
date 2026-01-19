import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from latency_encoding import latency_encode
from poisson_encoding import poisson_encode



# --- Processing Logic ---

def process_and_generate(source_root, output_root, encoding_type, T):
    print(f"\n>>> Generating {encoding_type}_{T} dataset...")
    
    source_root = Path(source_root)
    output_root = Path(output_root) / f"{encoding_type}_{T}"
    
    # Image extensions to search for
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Iterate through splits: train, val, test
    for split in ['train', 'val', 'test']:
        split_src = source_root / split
        split_dst = output_root / split
        
        if not split_src.exists():
            continue
            
        # 1. Handle Labels (Copy them)
        src_labels = split_src / "labels"
        dst_labels = split_dst / "labels"
        if src_labels.exists():
            if dst_labels.exists():
                shutil.rmtree(dst_labels)
            shutil.copytree(src_labels, dst_labels)
            
        # 2. Handle Images (Encode them)
        src_images_dir = split_src / "images"
        dst_images_dir = split_dst / "images"
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = [p for p in src_images_dir.glob('*') if p.suffix.lower() in image_extensions]
        
        for img_path in tqdm(image_files, desc=f"  {split}"):
            dest_pt_path = dst_images_dir / img_path.with_suffix('.pt').name
            
            # Skip if already exists
            if dest_pt_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use fixed size if needed, or keep original. 
            # Note: SpikeYOLO usually uses 640 or 1024. 
            # We'll keep original for flexibility, dataloader handles resizing.
            
            if encoding_type == 'poisson':
                spikes = poisson_encode(img, T)
            else:
                spikes = latency_encode(img, T)
                
            torch.save(spikes, dest_pt_path)
            
    # 3. Copy data.yaml and adjust paths if necessary
    src_yaml = source_root / "data.yaml"
    if src_yaml.exists():
        dst_yaml = output_root / "data.yaml"
        shutil.copy(src_yaml, dst_yaml)
        # Optional: You might want to edit dst_yaml to point correctly, 
        # but usually relative paths in yaml work if the structure is identical.

if __name__ == "__main__":
    # Get the directory where this script is located
    current_dir = Path(__file__).resolve().parent
    
    SOURCE_DATASET = current_dir.parent / "datasets" / "D-Fire"
    OUTPUT_BASE = current_dir.parent / "datasets"
    
    T_VALUES = [4, 6, 8]
    ENCODING_TYPES = ['poisson', 'latency']
    
    for T in T_VALUES:
        for enc in ENCODING_TYPES:
            process_and_generate(SOURCE_DATASET, OUTPUT_BASE, enc, T)
            
    print("\nAll datasets generated successfully in:", OUTPUT_BASE)
