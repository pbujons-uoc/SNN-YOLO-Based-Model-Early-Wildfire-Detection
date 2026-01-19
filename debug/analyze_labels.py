
import os
import glob
import numpy as np

# Path based on typical YOLO structure or what we find
base_dir = r"c:/Users/paubu/Desktop/Master/Setembre 2025 - Gener 2026/TFM/Code/datasets/D-Fire/train"
label_dir = os.path.join(base_dir, "labels")

if not os.path.exists(label_dir):
    print(f"Labels not found at {label_dir}. Checking parent/labels/train convention...")
    # Check parallel labels dir
    label_dir = r"c:/Users/paubu/Desktop/Master/Setembre 2025 - Gener 2026/TFM/Code/datasets/D-Fire/labels/train"

print(f"Checking labels in: {label_dir}")
if not os.path.exists(label_dir):
    print("Error: Could not find label directory.")
    exit(1)

files = glob.glob(os.path.join(label_dir, "*.txt"))
print(f"Found {len(files)} label files.")

counts = []
for f in files:
    with open(f, 'r') as fp:
        lines = [l.strip() for l in fp.readlines() if l.strip()]
        counts.append(len(lines))

if counts:
    counts = np.array(counts)
    print(f"Min instances: {counts.min()}")
    print(f"Max instances: {counts.max()}")
    print(f"Mean instances: {counts.mean():.2f}")
    print(f"Median instances: {np.median(counts)}")
    print(f"95th Percentile: {np.percentile(counts, 95)}")
    
    unique, ucounts = np.unique(counts, return_counts=True)
    print("\nInstance count distribution (Top 10):")
    # Sort by frequency
    sorted_indices = np.argsort(-ucounts)
    for i in sorted_indices[:10]:
        print(f"{unique[i]} instances: {ucounts[i]} images")
else:
    print("No labels found.")
