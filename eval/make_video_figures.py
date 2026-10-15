#!/usr/bin/env python3
"""Builds, from the outputs of predict_video.py (no GPU, no torch), for every video:
  <out>/<video>/side_by_side.mp4      the models in a grid, same frame, so they can be compared
  <out>/<video>/frames.pdf / .png     K evenly spaced frames x models (the figure for the paper)
  <out>/video_summary.csv             per video and model: frames, % of frames with >=1 smoke / fire detection, mean boxes per frame

    python make_video_figures.py --models yolov8m_s0 spikeyolo_rgb_T4_s0 spikeyolo_poisson_T4_s0 spikeyolo_latency_T4_s0 --frames 4
The percentages are descriptive (how often a model fires on a video); there is no ground truth, so they are NOT accuracy.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "Results_test").exists() else HERE.parent
LABEL = {"yolov8m_s0": "YOLOv8m", "spikeyolo_rgb_T4_s0": "SpikeYOLO direct", "spikeyolo_poisson_T4_s0": "SpikeYOLO Poisson",
         "spikeyolo_latency_T4_s0": "SpikeYOLO latency", "spikeyolo_rgb_T6_s0": "SpikeYOLO direct (T=6)",
         "spikeyolo_poisson_T6_s0": "SpikeYOLO Poisson (T=6)", "spikeyolo_latency_T6_s0": "SpikeYOLO latency (T=6)",
         "spikeyolo_rgb_T4_s0_window": "SpikeYOLO direct, window", "spikeyolo_poisson_T4_s0_window": "SpikeYOLO Poisson, window",
         "vanillacnn_long_s0": "VanillaCNN (180 ep.)"}


def read_frames(path, idx):
    cap = cv2.VideoCapture(str(path)); out = {}
    for k in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if ok:
            out[k] = f[:, :, ::-1]
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "Results_test" / "videos"))
    ap.add_argument("--models", nargs="+", default=["yolov8m_s0", "spikeyolo_rgb_T4_s0", "spikeyolo_poisson_T4_s0", "spikeyolo_latency_T4_s0"])
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--crop", nargs="*", default=[], metavar="VIDEO=START:END", help="use only frames START..END of a video (e.g. big_fire1=0:1200 to drop an end card), for the figure frames and the percentages")
    ap.add_argument("--width", type=int, default=640, help="width of each tile in the side-by-side video")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(a.out)
    crop = {}
    for c in a.crop:
        k, v = c.split('='); st, en = v.split(':'); crop[k] = (int(st), int(en))
    rows = []
    for vd in sorted(p for p in out.iterdir() if p.is_dir()):
        ms = [m for m in a.models if (vd / m / "annotated.mp4").exists()]
        if not ms:
            continue
        print(vd.name, ms)
        info = {}
        for m in ms:
            d = np.load(vd / m / "detections.npz")
            ptr, dets = d["ptr"], d["dets"]
            n = len(ptr) - 1
            st, en = crop.get(vd.name, (0, n)); en = min(en, n)
            cls = [dets[ptr[i]:ptr[i + 1], 5].astype(int) for i in range(st, en)]
            info[m] = (st, en)
            rows.append(dict(video=vd.name, model=m, frames=en - st, first_frame=st, last_frame=en - 1, threshold=float(d["threshold"]),
                             pct_frames_smoke=100 * np.mean([(c == 0).any() for c in cls]), pct_frames_fire=100 * np.mean([(c == 1).any() for c in cls]),
                             mean_boxes_per_frame=float((ptr[st + 1:en + 1] - ptr[st:en]).mean()),
                             **{k: (json.loads((vd / m / 'timing.json').read_text())[k] if (vd / m / 'timing.json').exists() else '') for k in ('ms_per_frame_mean', 'mJ_per_frame_net_mean')}))
        st0 = max(v[0] for v in info.values()); en0 = min(v[1] for v in info.values())
        n = min(en0, min(len(np.load(vd / m / 'detections.npz')['ptr']) - 1 for m in ms))
        # frames figure
        idx = np.linspace(st0 + (en0 - st0) * 0.1, st0 + (en0 - st0) * 0.9, a.frames).astype(int)
        fig, axs = plt.subplots(len(ms), a.frames, figsize=(a.frames * 3.0, len(ms) * 1.9), squeeze=False)
        for r, m in enumerate(ms):
            fr = read_frames(vd / m / "annotated.mp4", idx)
            for c, k in enumerate(idx):
                ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
                if k in fr:
                    h = fr[k]; h = h[int(h.shape[0] * 0.075):]            # crop the label banner
                    ax.imshow(h)
                if c == 0:
                    ax.set_ylabel(LABEL.get(m, m), fontsize=7)
                if r == 0:
                    ax.set_title(f"frame {k}", fontsize=7)
        plt.tight_layout(); fig.savefig(vd / "frames.pdf"); fig.savefig(vd / "frames.png", dpi=150); plt.close()
        # side by side video
        caps = [cv2.VideoCapture(str(vd / m / "annotated.mp4")) for m in ms]
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        w = a.width; hh = int(round(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT) * w / caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))) // 2 * 2
        cols = 2 if len(ms) > 1 else 1; rws = (len(ms) + cols - 1) // cols
        wr = cv2.VideoWriter(str(vd / "side_by_side.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cols * w, rws * hh))
        for _ in range(n):
            tiles = []
            for cp in caps:
                ok, f = cp.read()
                tiles.append(cv2.resize(f, (w, hh)) if ok else np.full((hh, w, 3), 255, np.uint8))
            while len(tiles) < cols * rws:
                tiles.append(np.full((hh, w, 3), 255, np.uint8))
            wr.write(np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rws)]))
        wr.release(); [c.release() for c in caps]
    if rows:
        with open(out / "video_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("\nvideo model frames %smoke %fire boxes/frame")
        for r in rows:
            print(f"{r['video']:28s} {LABEL.get(r['model'], r['model']):24s} {r['frames']:6d} {r['pct_frames_smoke']:6.1f} {r['pct_frames_fire']:6.1f} {r['mean_boxes_per_frame']:6.2f}")
        print("saved", out / "video_summary.csv")


if __name__ == "__main__":
    main()