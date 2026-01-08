import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the results file
results_path = os.path.join("Results", "YOLO", "results_gpu_papa", "results.csv")

if not os.path.exists(results_path):
    print(f"Error: Could not find results file at {results_path}")
    exit(1)

# Read the CSV
# The CSV often has spaces in column names, so we strip them
df = pd.read_csv(results_path)
df.columns = df.columns.str.strip()

print(f"Loaded results from {results_path}")
print("Available columns:", df.columns.tolist())

# Create a figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('YOLO Training Results', fontsize=16)

# 1. Box Loss
axs[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
axs[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
axs[0, 0].set_title('Box Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Class Loss (if available)
if 'train/cls_loss' in df.columns:
    axs[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
    axs[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
    axs[0, 1].set_title('Class Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
else:
    axs[0, 1].text(0.5, 0.5, 'Class Loss not available', ha='center', va='center')

# 3. mAP
axs[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (B)')
axs[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (B)')
axs[1, 0].set_title('mAP metrics')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('mAP')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4. Precision & Recall
axs[1, 1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision (B)')
axs[1, 1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall (B)')
axs[1, 1].set_title('Precision & Recall')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Score')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Layout adjustment
plt.tight_layout()

# Save the plot
output_file = 'yolo_training_visualization.png'
plt.savefig(output_file)
print(f"Visualization saved to {output_file}")
