#!/usr/bin/env python3
"""Run one trained model on the validation or test split and save every prediction to disk (the same
pipeline for every model). metrics.py then computes all the numbers from these files, with no GPU.

For a model whose input is random (Poisson) the pass is repeated with several generator seeds; deterministic
models (direct RGB, latency, YOLOv8m, VanillaCNN) are run once, since repeating gives identical outputs.

    python dump_predictions.py --model spikeyolo_poisson_T4_s0 --split test --seeds 5
    python dump_predictions.py --model yolov8m_s0 --split val

Output: ../Results_test/<model>/<split>_seed<k>.npz  (+ .json with the settings and environment)
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import models as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run name in Results_train, e.g. spikeyolo_rgb_T4_s0")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--seeds", type=int, default=1, help="number of generator seeds (used only for Poisson input)")
    ap.add_argument("--batch", type=int, default=8, help="batch size; keep it fixed: Poisson noise depends on the batch shape")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    ap.add_argument("--no-multi-label", action="store_true", help="single-label NMS (default: multi-label as the Ultralytics validator)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(M.RESULTS_TEST))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    spec = M.get_spec(a.model)
    seeds = list(range(a.seeds)) if spec.stochastic else [0]
    out_dir = Path(a.out) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in seeds if a.force or not (out_dir / f"{a.split}_seed{s}.npz").exists()]
    if not todo:
        print(f"[skip] {spec.name} {a.split}: all seeds already dumped")
        return

    device = M.device_from(a.device)
    L = M.load(spec, device, multi_label=not a.no_multi_label)
    loader, names = M.build_loader(L.U, a.split, a.batch, a.workers)
    imgsz = M.IMGSZ
    print(f"[{spec.name}] {spec.family} params {L.params / 1e6:.2f}M  ultralytics {L.U.version}  split {a.split}  seeds {todo}")

    for seed in todo:
        if L.layer0 is not None and hasattr(L.layer0, "seed"):
            L.layer0.seed = seed                      # fixed generator seed of the Poisson encoder (eval mode)
        files, dets, gts, det_ptr, gt_ptr = [], [], [], [0], [0]
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["img"].to(device, non_blocking=True).float() / 255
                with torch.autocast("cuda", dtype=torch.float16, enabled=(a.precision == "amp" and device.type == "cuda")):
                    out = L.net(imgs)
                preds = L.decode(out)
                for si, det in enumerate(preds):
                    idx = batch["batch_idx"] == si
                    cls = batch["cls"][idx].view(-1, 1).float()
                    box = L.U.ops.xywh2xyxy(batch["bboxes"][idx]) * imgsz
                    g = torch.cat([cls, box], 1).cpu().numpy().astype(np.float32).reshape(-1, 5)
                    d = det.float().cpu().numpy().astype(np.float32).reshape(-1, 6)
                    files.append(batch["im_file"][si]); dets.append(d); gts.append(g)
                    det_ptr.append(det_ptr[-1] + len(d)); gt_ptr.append(gt_ptr[-1] + len(g))
        path = out_dir / f"{a.split}_seed{seed}.npz"
        np.savez_compressed(path, files=np.array(files), det_ptr=np.array(det_ptr, dtype=np.int64),
                            dets=np.concatenate(dets).reshape(-1, 6), gt_ptr=np.array(gt_ptr, dtype=np.int64),
                            gts=np.concatenate(gts).reshape(-1, 5))
        meta = dict(model=spec.name, family=spec.family, T=spec.T, encoding=spec.encoding, split=a.split, seed=seed,
                    n_images=len(files), n_detections=int(det_ptr[-1]), n_gt=int(gt_ptr[-1]), batch=a.batch,
                    precision=a.precision, conf=0.001, iou=0.7, max_det=300, multi_label=not a.no_multi_label,
                    imgsz=imgsz, weights=str(spec.weights), params=L.params, ultralytics=L.U.version,
                    torch=torch.__version__, python=platform.python_version(), host=platform.node(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    seconds=time.time() - t0, names=names if isinstance(names, dict) else dict(enumerate(names)))
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"  seed {seed}: {len(files)} images, {int(det_ptr[-1])} detections -> {path}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
