#!/usr/bin/env python3
"""Collect the per-epoch metrics of every run in Results_train and make CSVs + figures.

    python collect_and_plot.py --results ../Results_train

Reads  <results>/<run>/epoch_metrics.csv  (falls back to Ultralytics' results.csv).
Run names: spikeyolo_<rgb|poisson|latency>_T<T>_s<seed>; any other run (yolov8..., vanillacnn...)
is treated as a baseline and drawn as a grey dashed curve.

Writes in <results>/summary/ :
    all_runs_epoch_metrics.csv   one row per run and epoch (long format, ready for any plotting tool)
    run_summary.csv              one row per run (best epoch, final values, time, memory)
and figures (PNG + PDF) in <results>/figures/ .
"""
import argparse
import os
import re

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAME_RE = re.compile(r"^(?P<model>spikeyolo)_(?P<encoding>rgb|poisson|latency)_T(?P<T>\d+)_s(?P<seed>\d+)(?:_quick)?$")
COLORS = {"rgb": "#0072B2", "poisson": "#D55E00", "latency": "#009E73"}
LABELS = {"rgb": "Direct RGB", "poisson": "Poisson", "latency": "Latency"}
RESULTS_CSV_MAP = {  # Ultralytics results.csv -> our names
    "metrics/precision(B)": "precision", "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "map50", "metrics/mAP50-95(B)": "map50_95",
    "train/box_loss": "train_box_loss", "train/cls_loss": "train_cls_loss", "train/dfl_loss": "train_dfl_loss",
    "val/box_loss": "val_box_loss", "val/cls_loss": "val_cls_loss", "val/dfl_loss": "val_dfl_loss",
    "lr/pg0": "lr_pg0", "lr/pg1": "lr_pg1", "lr/pg2": "lr_pg2",
}
COMPARABLE = {"map50", "map50_95", "precision", "recall", "fire_ap50", "smoke_ap50", "epoch_seconds"}
# YOLOv8m uses the same Ultralytics detection loss (box / cls / dfl) as SpikeYOLO, so its loss curves are drawn too
# (different Ultralytics version: 8.3.231 vs the 8.0.197 fork). VanillaCNN has no DFL and its "cls" loss includes
# objectness, so its losses are never drawn with the others.
LOSS_COLS = {"train_box_loss", "train_cls_loss", "train_dfl_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss"}
BASE_STYLE = {"yolov8": dict(color="0.35", ls="--", lw=1.4), "vanilla": dict(color="0.6", ls=":", lw=1.6)}
CURVES = [  # (column, y label, file stem)
    ("map50", "mAP@0.5 (validation)", "map50"),
    ("map50_95", "mAP@0.5:0.95 (validation)", "map50_95"),
    ("precision", "Precision (validation)", "precision"),
    ("recall", "Recall (validation)", "recall"),
    ("train_box_loss", "Train box loss", "train_box_loss"),
    ("train_cls_loss", "Train cls loss", "train_cls_loss"),
    ("train_dfl_loss", "Train dfl loss", "train_dfl_loss"),
    ("val_box_loss", "Validation box loss", "val_box_loss"),
    ("val_cls_loss", "Validation cls loss", "val_cls_loss"),
    ("val_dfl_loss", "Validation dfl loss", "val_dfl_loss"),
    ("fire_ap50", "Fire AP@0.5 (validation)", "fire_ap50"),
    ("smoke_ap50", "Smoke AP@0.5 (validation)", "smoke_ap50"),
]


def load_runs(root):
    frames = []
    for run in sorted(os.listdir(root)):
        d = os.path.join(root, run)
        if not os.path.isdir(d) or run.startswith("_") or run in ("summary", "figures"):
            continue
        p1, p2 = os.path.join(d, "epoch_metrics.csv"), os.path.join(d, "results.csv")
        if os.path.isfile(p1):
            df = pd.read_csv(p1)
        elif os.path.isfile(p2):
            df = pd.read_csv(p2)
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns=RESULTS_CSV_MAP)
        else:
            continue
        m = NAME_RE.match(run)
        df.insert(0, "run", run)
        df.insert(1, "model", m["model"] if m else run)
        df.insert(2, "encoding", m["encoding"] if m else "")
        df.insert(3, "T", int(m["T"]) if m else np.nan)
        df.insert(4, "seed", int(m["seed"]) if m else np.nan)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no epoch_metrics.csv / results.csv found under {root}")
    return pd.concat(frames, ignore_index=True)


def summarize(df):
    rows = []
    for run, g in df.groupby("run", sort=False):
        score = g["fitness"] if "fitness" in g and g["fitness"].notna().any() else 0.1 * g["map50"] + 0.9 * g["map50_95"]
        b = g.loc[score.idxmax()]
        r = {"run": run, "model": g["model"].iloc[0], "encoding": g["encoding"].iloc[0], "T": g["T"].iloc[0],
             "seed": g["seed"].iloc[0], "epochs_done": int(g["epoch"].max()), "best_epoch": int(b["epoch"])}
        for c in ("map50", "map50_95", "precision", "recall", "fire_ap50", "fire_ap50_95", "smoke_ap50", "smoke_ap50_95"):
            if c in g:
                r[f"best_{c}"] = b[c]
                r[f"last_{c}"] = g[c].iloc[-1]
        if "epoch_seconds" in g:
            r["mean_epoch_seconds"] = g["epoch_seconds"].mean()
            r["total_train_hours"] = g["epoch_seconds"].sum(min_count=1) / 3600
        if "gpu_mem_gb" in g:
            r["max_gpu_mem_gb"] = g["gpu_mem_gb"].max()
        rows.append(r)
    return pd.DataFrame(rows)


def save(fig, stem, outdir):
    fig.savefig(os.path.join(outdir, stem + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(outdir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(df, col, ylabel, stem, outdir):
    if col not in df or df[col].notna().sum() == 0:
        return
    spike = df[df["encoding"] != ""]
    base = df[df["encoding"] == ""]
    Ts = sorted(spike["T"].dropna().unique())
    if not Ts:
        return
    fig, axes = plt.subplots(1, len(Ts), figsize=(4.2 * len(Ts), 3.4), sharey=True, squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for enc in ("rgb", "poisson", "latency"):
            for _, g in spike[(spike["T"] == T) & (spike["encoding"] == enc)].groupby("run"):
                ax.plot(g["epoch"], g[col], color=COLORS[enc], marker="o", ms=3, lw=1.6, label=LABELS[enc])
        for run, g in base.groupby("run"):
            if run.endswith("_long_s0") or "_long" in run:                      # drawn in its own figure (180 epochs)
                continue
            fam = "yolov8" if run.startswith("yolov8") else "vanilla"
            if not (col in COMPARABLE or (col in LOSS_COLS and fam == "yolov8")):
                continue
            ax.plot(g["epoch"], g[col], label=("YOLOv8m" if fam == "yolov8" else "VanillaCNN (25 epochs)"), **BASE_STYLE[fam])
        ax.set_title(f"T = {int(T)}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen), frameon=False, bbox_to_anchor=(0.5, 1.08))
    save(fig, "curves_" + stem, outdir)


def plot_loss_comparison(df, T, outdir):
    """One figure per T, like the old 'Loss Comparison During Training': rows = train / validation, columns = box / cls /
    dfl; SpikeYOLO (direct, Poisson, latency at this T), YOLOv8m and VanillaCNN together.
    VanillaCNN has no DFL term and its cls loss includes objectness, so its curves are on another scale (dotted)."""
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), squeeze=False)
    for r, split in enumerate(("train", "val")):
        for c, kind in enumerate(("box", "cls", "dfl")):
            ax, col = axes[r][c], f"{split}_{kind}_loss"
            if col in df:
                for enc in ("rgb", "poisson", "latency"):
                    for _, g in df[(df["T"] == T) & (df["encoding"] == enc)].groupby("run"):
                        ax.plot(g["epoch"], g[col], color=COLORS[enc], lw=1.6, label=f"SpikeYOLO {LABELS[enc]}")
                for run, g in df[df["encoding"] == ""].groupby("run"):
                    if "_long" in run or g[col].notna().sum() == 0:
                        continue
                    fam = "yolov8" if run.startswith("yolov8") else "vanilla"
                    ax.plot(g["epoch"], g[col], label=("YOLOv8m" if fam == "yolov8" else "VanillaCNN (own loss)"), **BASE_STYLE[fam])
            ax.set_title(f"{'Train' if split == 'train' else 'Validation'} {kind} loss", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    h, l = axes[0][0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=len(seen), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(f"Loss comparison during training (SpikeYOLO T = {int(T)})", y=1.08, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, f"loss_comparison_T{int(T)}", outdir)


def plot_vanilla_convergence(df, summ, outdir):
    """VanillaCNN at 25 and at 180 epochs, with YOLOv8m and SpikeYOLO RGB T=4 (best values) as references."""
    runs = [r for r in ("vanillacnn_s0", "vanillacnn_long_s0") if r in set(df["run"])]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, (col, yl) in zip(axes, (("map50", "Validation mAP@0.5"), ("map50_95", "Validation mAP@0.5:0.95"))):
        for r, c, lab in (("vanillacnn_s0", "0.6", "VanillaCNN, 25 epochs"), ("vanillacnn_long_s0", "#CC79A7", "VanillaCNN, 180 epochs")):
            g = df[df["run"] == r]
            if len(g):
                ax.plot(g["epoch"], g[col], color=c, lw=1.4, label=lab)
        for r, lab, st in (("yolov8m_s0", "YOLOv8m (best)", dict(color="0.35", ls="--")),
                           ("spikeyolo_rgb_T4_s0", "SpikeYOLO RGB, T=4 (best)", dict(color=COLORS["rgb"], ls="-."))):
            row = summ[summ["run"] == r]
            if len(row):
                ax.axhline(float(row["best_" + col].iloc[0]), lw=1.2, label=lab, **st)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 1.1), fontsize=8)
    fig.tight_layout()
    save(fig, "curves_vanillacnn_convergence", outdir)


def plot_vs_T(summ, cols, ylabel, stem, outdir):
    s = summ[summ["encoding"] != ""]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for enc in ("rgb", "poisson", "latency"):
        g = s[s["encoding"] == enc].groupby("T")[cols].mean().sort_index()
        if len(g):
            ax.plot(g.index, g.values, color=COLORS[enc], marker="o", lw=1.8, label=LABELS[enc])
    for _, r in summ[summ["encoding"] == ""].iterrows():
        if cols in r and pd.notna(r[cols]):
            ax.axhline(r[cols], color="0.5", ls="--", lw=1.2, label=r["run"])
    ax.set_xlabel("Time steps T")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    save(fig, stem, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../Results_train")
    a = ap.parse_args()
    sd, fd = os.path.join(a.results, "summary"), os.path.join(a.results, "figures")
    os.makedirs(sd, exist_ok=True)
    os.makedirs(fd, exist_ok=True)
    df = load_runs(a.results)
    summ = summarize(df)
    df.to_csv(os.path.join(sd, "all_runs_epoch_metrics.csv"), index=False)
    summ.to_csv(os.path.join(sd, "run_summary.csv"), index=False)
    for col, yl, stem in CURVES:
        plot_curves(df, col, yl, stem, fd)
    plot_curves(df, "epoch_seconds", "Seconds per epoch", "epoch_seconds", fd)
    plot_vanilla_convergence(df, summ, fd)
    for T in sorted(df.loc[df["encoding"] != "", "T"].dropna().unique()):
        plot_loss_comparison(df, T, fd)
    plot_vs_T(summ, "best_map50", "Best validation mAP@0.5", "best_map50_vs_T", fd)
    plot_vs_T(summ, "best_map50_95", "Best validation mAP@0.5:0.95", "best_map50_95_vs_T", fd)
    if "mean_epoch_seconds" in summ:
        plot_vs_T(summ, "mean_epoch_seconds", "Mean seconds per epoch", "epoch_time_vs_T", fd)
    if "max_gpu_mem_gb" in summ:
        plot_vs_T(summ, "max_gpu_mem_gb", "Peak GPU memory (GB)", "gpu_mem_vs_T", fd)
    print(f"{df['run'].nunique()} runs -> {sd} and {fd}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()