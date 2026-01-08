import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths
paths = {
    "YOLO": os.path.join("Results", "YOLO", "results_gpu_papa", "results.csv"),
    "SpikeYOLO": os.path.join("Results", "SpikeYOLO", "results_gpu_papa", "results.csv"),
    "VanillaCNN": os.path.join("Results", "VanillaCNN", "D-Fire", "vanilla_cnn_results.csv")
}

# Load data
dfs = {}
for name, path in paths.items():
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            dfs[name] = df
            print(f"Loaded {name} with {len(df)} epochs.")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    else:
        print(f"File not found for {name}: {path}")

# Setup figure
# Layout:
# 1. mAP50 Comparison (All in one)
# 2. Training Loss (Subplots per model)
# 3. Validation Loss (Subplots per model where available)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3)

# --- 1. mAP50 Comparison ---
ax_map = fig.add_subplot(gs[0, :])  # Span all columns in first row
ax_map.set_title("Validation mAP50 Comparison", fontsize=14)
ax_map.set_xlabel("Epoch")
ax_map.set_ylabel("mAP50")
ax_map.grid(True)

for name, df in dfs.items():
    if name == "YOLO" or name == "SpikeYOLO":
        if 'metrics/mAP50(B)' in df.columns:
            ax_map.plot(df['epoch'], df['metrics/mAP50(B)'], label=f"{name} mAP50", linewidth=2)
    elif name == "VanillaCNN":
        if 'mAP50' in df.columns:
            ax_map.plot(df['epoch'], df['mAP50'], label=f"{name} mAP50", linewidth=2, linestyle='--')

ax_map.legend()

# --- 2. Loss Plots (Row 2) ---
# YOLO Loss
ax_loss_yolo = fig.add_subplot(gs[1, 0])
if "YOLO" in dfs:
    df = dfs["YOLO"]
    columns_to_plot = ['train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss']
    for col in columns_to_plot:
        if col in df.columns:
            ax_loss_yolo.plot(df['epoch'], df[col], label=col)
    ax_loss_yolo.set_title("YOLO Losses")
    ax_loss_yolo.set_xlabel("Epoch")
    ax_loss_yolo.set_ylabel("Loss")
    ax_loss_yolo.legend()
    ax_loss_yolo.grid(True)

# SpikeYOLO Loss
ax_loss_spike = fig.add_subplot(gs[1, 1])
if "SpikeYOLO" in dfs:
    df = dfs["SpikeYOLO"]
    columns_to_plot = ['train/box_loss', 'train/cls_loss', 'val/box_loss', 'val/cls_loss']
    for col in columns_to_plot:
        if col in df.columns:
            ax_loss_spike.plot(df['epoch'], df[col], label=col)
    ax_loss_spike.set_title("SpikeYOLO Losses")
    ax_loss_spike.set_xlabel("Epoch")
    ax_loss_spike.legend()
    ax_loss_spike.grid(True)

# VanillaCNN Loss
ax_loss_vanilla = fig.add_subplot(gs[1, 2])
if "VanillaCNN" in dfs:
    df = dfs["VanillaCNN"]
    if 'train_loss' in df.columns:
        ax_loss_vanilla.plot(df['epoch'], df['train_loss'], label='Train Loss', color='green')
    ax_loss_vanilla.set_title("VanillaCNN Loss")
    ax_loss_vanilla.set_xlabel("Epoch")
    ax_loss_vanilla.legend()
    ax_loss_vanilla.grid(True)

plt.tight_layout()
output_file = 'models_comparison.png'
plt.savefig(output_file)
print(f"Comparison plot saved to {output_file}")
