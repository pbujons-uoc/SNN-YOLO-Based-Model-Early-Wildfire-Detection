#!/usr/bin/env python3
"""Qualitative figure: 12 test images (3 empty, 3 smoke only, 3 fire only, 3 fire+smoke) x
{ground truth, YOLOv8m, SpikeYOLO direct / Poisson / latency (T=4)}. Reads the dumps of the test evaluation (no GPU).

    python make_qualitative.py                      # a new random set of 12 images every run (seed printed)
    python make_qualitative.py --seed 3             # optional: reproducible draw
    python make_qualitative.py --exclude ../Results_test/leakage_strict/leakage_test_confirmed.txt   # only images WITHOUT near-duplicates
    python make_qualitative.py --files a.jpg b.jpg ...   # or give the 12 file names yourself (the ones of your earlier 12-image subset)
Output: ../Results_test/summary/qualitative_12.pdf / .png   (copy the pdf to images/ in the manuscript)
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metrics as MT

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "Results_test").exists() else HERE.parent
MODELS = [("yolov8m_s0", "YOLOv8m"), ("spikeyolo_rgb_T4_s0", "SpikeYOLO direct"),
          ("spikeyolo_poisson_T4_s0", "SpikeYOLO Poisson"), ("spikeyolo_latency_T4_s0", "SpikeYOLO latency")]
COL = {0: "#1f77b4", 1: "#d62728"}      # smoke blue, fire red  (class 0 smoke, 1 fire)


def letterbox(img, size=640):
    w, h = img.size
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    im = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def draw(ax, img, boxes, cls):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for b, c in zip(boxes, cls):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=COL[int(c)], lw=1.4))
    for s in ax.spines.values():
        s.set_linewidth(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "Results_test"))
    ap.add_argument("--images", default=str(ROOT / "datasets" / "D-Fire_clean" / "test" / "images"))
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--exclude", default=None, help="text file with test images to leave out (leakage_test_confirmed.txt: images with a near-duplicate in train/val)")
    ap.add_argument("--seed", type=int, default=None, help="default: a new random draw every run (the seed used is printed)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = Path(a.results)
    dumps = {m: MT.Dump(res / m / "test_seed0.npz") for m, _ in MODELS}
    thr = {m: json.loads((res / m / "metrics_seed0.json").read_text())["threshold"] for m, _ in MODELS}
    ref = dumps[MODELS[0][0]]
    names = [Path(str(f)).name for f in ref.files]
    if a.files:
        idx = [names.index(Path(f).name) for f in a.files]
    else:
        seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        print('random seed used:', seed)
        rng = np.random.default_rng(seed)
        cl = [set(ref.gt(i)[:, 0].astype(int).tolist()) for i in range(ref.n)]
        banned = set()
        if a.exclude:
            banned = {Path(l.strip()).name for l in Path(a.exclude).read_text().splitlines() if l.strip()}
            print(f"excluding {len(banned)} test images with a near-duplicate in train/val")
        ok = [names[i] not in banned for i in range(ref.n)]
        groups = [[i for i in range(ref.n) if ok[i] and cl[i] == s_] for s_ in (set(), {0}, {1}, {0, 1})]
        print("candidates per group (empty, smoke, fire, both):", [len(g) for g in groups])
        if min(len(g) for g in groups) < 3:
            raise SystemExit("fewer than 3 candidates in a group; use a smaller exclusion list")
        idx = [int(i) for g in groups for i in rng.choice(g, 3, replace=False)]
    print("images used (pass them with --files to reproduce):")
    for i in idx:
        print("  ", names[i], "GT classes:", sorted(set(ref.gt(i)[:, 0].astype(int).tolist())))
    ncols, nrows = 6, 2 * (1 + len(MODELS))                 # two blocks of 6 images, each block has GT + 4 models
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    for k, i in enumerate(idx):
        blk, col = divmod(k, ncols)
        p = next(Path(a.images).glob(Path(names[i]).stem + ".*"))
        img = letterbox(Image.open(p).convert("RGB"))
        g = ref.gt(i)
        draw(axs[blk * 5, col], img, g[:, 1:5], g[:, 0])
        for r, (m, label) in enumerate(MODELS):
            d = dumps[m].det(i); d = d[d[:, 4] >= thr[m]]
            draw(axs[blk * 5 + 1 + r, col], img, d[:, :4], d[:, 5])
        for r in range(5):
            if col == 0:
                axs[blk * 5 + r, 0].set_ylabel((["Ground truth"] + [l for _, l in MODELS])[r], fontsize=7)
    for ax in axs.ravel():
        ax.xaxis.set_visible(False)
    fig.legend(handles=[plt.Line2D([], [], color=COL[0], lw=2), plt.Line2D([], [], color=COL[1], lw=2)], labels=["smoke", "fire"],
               loc="upper center", ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out) if a.out else res / "summary" / "qualitative_12"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=150)
    print("saved", str(out) + ".pdf")


if __name__ == "__main__":
    main()