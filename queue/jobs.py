"""Experiment queue. Edit the constants, enable the jobs whose scripts are ready, then run:

    python run_queue.py --dry-run
    tmux new -s train      (or nohup)
    python run_queue.py
"""
import os

# QUICK=1 python run_queue.py --state queue_state_quick.json   -> smoke test of the whole queue:
# 2 epochs on 5% of the training images for every job, results in ../Results_quick, job names end in _quick.
QUICK = os.environ.get("QUICK") == "1"
SUFFIX = "_quick" if QUICK else ""
EXTRA = ["--fraction", "0.05"] if QUICK else []

# ------------------------------------------------------------------ common recipe
EPOCHS = 2 if QUICK else 25              # same for every COCO-pretrained detector (SpikeYOLO rgb/poisson/latency, YOLOv8). With AMP, T=4 and T=6 trios at 25 epochs: ~36 h (estimate)
SEEDS = [0]              # add 1, 2 for repeated runs when time allows
TIMESTEPS = [4, 6]        # run order: T=4 first. Per epoch for the 3 inputs with AMP (estimate): T2 ~18 min, T4 ~35 min, T6 ~50 min, T8 ~67 min
VANILLA_LONG_EPOCHS = 2 if QUICK else 180 # VanillaCNN also trained to convergence (its detection head starts from scratch)
AMP = True               # passed the overfit test (same losses/mAP as fp32 within noise, 1.65x faster training, half the memory); applies to every SpikeYOLO run
BATCH = {2: 4, 4: 4, 6: 2, 8: 2}   # per-step batch that fits 32 GB without AMP (T*batch <= 16; measured 25.5 GB at T4 b4, 21.3 GB at T6 b2)
NBS = 64                 # effective batch (Ultralytics default; lr0=0.01 is paired with it). ~4,800 updates in 30 epochs
OPTIMIZER, LR0, LRF = "SGD", 0.01, 0.004   # the SpikeYOLO default recipe (ultralytics/cfg/default.yaml); tested on the overfit set
WARMUP = 3               # warmup epochs
CLOSE_MOSAIC = max(1, round(EPOCHS / 6)) if EPOCHS else 1   # mosaic off in the last sixth of training (same for every model)
PROJECT = "../Results_quick" if QUICK else "../Results_train"   # all results go to code/Results_train (relative to the model folder SpikeYOLO)
DATA = "../datasets/D-Fire_clean/data.yaml"
SPIKE_MODEL = "snn_yolov8s.yaml"   # scale s = SpikeYOLOs 23.1M (same scale for every SpikeYOLO variant)
SPIKE_WEIGHTS = "23M_best.pt"      # <-- put the real filename of the 23M README checkpoint
VANILLA_DIR = "../VanillaCNN"      # folder with train_vanillacnn.py, model.py, loss.py
YOLO_DIR = "../ultralytics"          # folder with train_yolov8.py and yolov8m.pt (official YOLOv8 package)
YOLO_WEIGHTS = "yolov8m.pt"
SPIKE_DIR = "../SpikeYOLO"         # folder with the SpikeYOLO training scripts (cwd of the jobs); run the queue from code/queue


def _spike_job(tag, script, extra, T, seed, enabled):
    name = f"{tag}_T{T}_s{seed}{SUFFIX}"
    run_dir = os.path.join(SPIKE_DIR, PROJECT, name)
    cmd = ["python", script, "--data", DATA, "--model", SPIKE_MODEL, "--weights", SPIKE_WEIGHTS,
           "--T", str(T), "--batch", str(BATCH[T]), "--close-mosaic", str(CLOSE_MOSAIC),
           "--nbs", str(NBS), "--warmup-epochs", str(WARMUP),
           "--optimizer", OPTIMIZER, "--lr0", str(LR0), "--lrf", str(LRF),
           "--epochs", str(EPOCHS), "--seed", str(seed), "--name", name,
           "--project", PROJECT] + (["--amp"] if AMP else []) + extra + EXTRA
    return dict(
        name=name,
        cmd=cmd,
        cwd=SPIKE_DIR,
        enabled=enabled,
        run_dir=run_dir,
        done_file=os.path.join(run_dir, "weights", "best.pt"),
        resume_args=["--resume"],
        resume_check=os.path.join(run_dir, "weights", "last.pt"),
    )


def _vanilla_job(seed, epochs, name, enabled=True):
    run_dir = os.path.join(VANILLA_DIR, PROJECT, name)
    cmd = ["python", "train_vanillacnn.py", "--data", DATA, "--epochs", str(epochs), "--seed", str(seed),
           "--close-mosaic", str(max(1, round(epochs / 6))), "--name", name, "--project", PROJECT] + EXTRA
    return dict(name=name, cmd=cmd, cwd=VANILLA_DIR, enabled=enabled, run_dir=run_dir,
                done_file=os.path.join(run_dir, "weights", "best.pt"),
                resume_args=["--resume"], resume_check=os.path.join(run_dir, "weights", "last.pt"))


def _yolo_job(seed):
    name = f"yolov8m_s{seed}{SUFFIX}"
    run_dir = os.path.join(YOLO_DIR, PROJECT, name)
    cmd = ["python", "train_yolov8.py", "--data", DATA, "--weights", YOLO_WEIGHTS, "--epochs", str(EPOCHS),
           "--batch", "16", "--close-mosaic", str(CLOSE_MOSAIC), "--nbs", str(NBS), "--warmup-epochs", str(WARMUP),
           "--optimizer", OPTIMIZER, "--lr0", str(LR0), "--lrf", str(LRF), "--seed", str(seed),
           "--name", name, "--project", PROJECT] + EXTRA
    return dict(name=name, cmd=cmd, cwd=YOLO_DIR, enabled=True, run_dir=run_dir,
                done_file=os.path.join(run_dir, "weights", "best.pt"),
                resume_args=["--resume"], resume_check=os.path.join(run_dir, "weights", "last.pt"))


JOBS = []
if EPOCHS is not None:
    for seed in SEEDS:
        JOBS.append(_vanilla_job(seed, EPOCHS, f"vanillacnn_s{seed}{SUFFIX}"))   # cheap baseline first: quick feedback on the pipeline
        JOBS.append(_yolo_job(seed))   # about 1-2 h: second job so the whole YOLO pipeline is checked early
        for T in TIMESTEPS:
            # the three SpikeYOLO variants: same script, same recipe; only --encoding (layer 0) changes
            JOBS.append(_spike_job("spikeyolo_rgb", "train_spikeyolo.py", ["--encoding", "rgb"], T, seed, enabled=True))
            JOBS.append(_spike_job("spikeyolo_poisson", "train_spikeyolo.py", ["--encoding", "poisson"], T, seed, enabled=True))
            JOBS.append(_spike_job("spikeyolo_latency", "train_spikeyolo.py", ["--encoding", "latency"], T, seed, enabled=True))
        # last: VanillaCNN trained to convergence, so it never delays the main comparison
        JOBS.append(_vanilla_job(seed, VANILLA_LONG_EPOCHS, f"vanillacnn_long_s{seed}{SUFFIX}"))