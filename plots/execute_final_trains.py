import subprocess
import os
import sys

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_EXE = sys.executable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FINAL_DIR = os.path.join(ROOT_DIR, "Results_train_final")

# Common Parameters
EPOCHS = 45
BATCH_SIZE = 4
IMGSZ = 640
LR = 0.001  # Default learning rate if not specified per experiment

# Model Configurations
'''
    
    {
        "name": "SpikeYOLO_poisson",
        "script": "SpikeYOLO_Encoded/train.py",
        "data": "datasets/poisson_4/data.yaml",
        "extra_args": [],
        "lr": 0.001
    },
    {
        "name": "Yolov8",
        # Uses the specific script we modified
        "script": "ultralytics/train_yolov8l.py",
        "data": "datasets/D-Fire/data.yaml",
        "extra_args": [],
        "lr": 0.001
    },
'''
EXPERIMENTS = [
    {
        "name": "SpikeYOLO_latency",
        "script": "SpikeYOLO_Encoded/train.py",
        "data": "datasets/latency_4/data.yaml",
        "extra_args": [],
        "lr": 0.001
    },
]

def run_experiment(exp):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {exp['name']}")
    print(f"{'='*60}")

    # Construct command
    # We pass the absolute path to the data yaml to avoid confusion
    data_path = os.path.join(ROOT_DIR, exp['data'])
    script_path = os.path.join(ROOT_DIR, exp['script'])

    # IMPORTANT: The 'project' and 'name' args determine the output folder structure.
    # We want: Results_train_final/[ModelName]
    # Ultralytics style: project=Results_train_final, name=[ModelName] -> creates Results_train_final/[ModelName]
    
    # Compute Epochs: VanillaCNN needs more time (from scratch)
    # Others use the global default (50)
    epochs_to_run = 180 if "VanillaCNN" in exp['name'] else EPOCHS

    # Use per-experiment LR if provided, otherwise global LR
    lr_to_use = exp.get('lr', LR)
    cmd = [
        PYTHON_EXE, script_path,
        "--data", data_path,
        "--epochs", str(epochs_to_run),
        "--batch", str(BATCH_SIZE),
        "--imgsz", str(IMGSZ),
        "--lr", str(lr_to_use),
        "--project", RESULTS_FINAL_DIR,
        "--name", exp['name']
    ]
    
    # Add extra args if any
    cmd.extend(exp['extra_args'])

    # Print LR info
    print(f"Using learning rate {lr_to_use} for experiment {exp['name']}")

    # For VanillaCNN, it might handle 'project'/'name' differently or default to 'runs'.
    # I checked VanillaCNN/train.py: it uses DEFAULT_RESULTS_ROOT = "runs" and creates subdir based on dataset.
    # But wait, looking at my previous view of VanillaCNN/train.py, it DOES NOT accept --project or --name.
    # It hardcodes save_dir = RESULTS_ROOT / dataset_name.
    # We need to be careful.
    if "VanillaCNN" in exp['name']:
        # Special handling for VanillaCNN if it ignores project/name
        # The script lets us pass args, but it doesn't utilize --project/--name in the saving logic shown.
        # It saves to "runs/[DatasetName]".
        # We might need to manually move the folder later or modify VanillaCNN/train.py.
        # Let's modify VanillaCNN/train.py to accept --project/--name or just handle the move.
        # Handling the move is risky.
        # I'll modify VanillaCNN/train.py separately to respect these args.
        pass

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[SUCCESS] Experiment {exp['name']} finished.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Experiment {exp['name']} failed with error code {e.returncode}.")
        # We continue to the next experiment? User said "execute sequentially". 
        # Usually better to stop or ask. But I'll continue to try others.
        print("Continuing to next experiment...")

def main():
    os.makedirs(RESULTS_FINAL_DIR, exist_ok=True)

    for exp in EXPERIMENTS:
        run_experiment(exp)

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED.")
    print(f"Results located in: {RESULTS_FINAL_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
