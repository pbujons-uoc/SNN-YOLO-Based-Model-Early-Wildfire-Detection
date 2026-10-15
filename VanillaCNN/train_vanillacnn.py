#!/usr/bin/env python3
"""Train and evaluate the VanillaCNN baseline (ResNet18 + 1x1 grid head, S x S, one box per cell) with the
same data pipeline and the same metric code as SpikeYOLO / YOLOv8.

What stays exactly as in the original VanillaCNN: model.py (SimpleYoloCNN), loss.py (SimpleYoloLoss),
Adam + cosine schedule, one box per grid cell.
What changes (to make the comparison fair):
  * data come from the SAME Ultralytics dataset/dataloader code used for SpikeYOLO (letterbox, mosaic,
    HSV, flips, same train/val splits). The grid targets are built from those batches, so the targets are in the
    same (letterboxed) frame as the image. The old data_to_yolo.py built them from the un-letterboxed labels
    and applied random ColorJitter also at validation time;
  * evaluation = Ultralytics' own NMS (conf 0.001, IoU 0.7, max 300 detections) and AP code (DetMetrics), the same
    functions the Ultralytics validator uses for SpikeYOLO / YOLOv8;
  * results go to <project>/<name>/epoch_metrics.csv in the same format as the other models.

Put this file in the VanillaCNN folder (next to model.py and loss.py). Run it from there:
    python train_vanillacnn.py --data ../datasets/D-Fire_clean/data.yaml --epochs 4 --name vanillacnn_s0
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import SimpleYoloCNN      # noqa: E402  (unchanged VanillaCNN)
from loss import SimpleYoloLoss      # noqa: E402  (unchanged VanillaCNN loss)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="data.yaml (D-Fire_clean)")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="../Results_train")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--S", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="ram")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--ultralytics-dir", default="../SpikeYOLO", help="folder that contains the ultralytics package used for SpikeYOLO")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class U:  # ultralytics pieces, filled by import_ultralytics()
    pass


def import_ultralytics(path):
    """Import the same ultralytics package SpikeYOLO uses, whatever else is installed."""
    path = os.path.abspath(path)
    sys.path.insert(0, path)
    try:
        import ultralytics
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import DEFAULT_CFG, ops
        from ultralytics.utils.metrics import DetMetrics, box_iou
        from ultralytics.utils.torch_utils import init_seeds
    finally:
        sys.path.remove(path)
    assert os.path.abspath(ultralytics.__file__).startswith(path), f"wrong ultralytics: {ultralytics.__file__}"
    for k, v in dict(version=ultralytics.__version__, get_cfg=get_cfg, build_dataloader=build_dataloader,
                     build_yolo_dataset=build_yolo_dataset, check_det_dataset=check_det_dataset, DEFAULT_CFG=DEFAULT_CFG,
                     ops=ops, DetMetrics=DetMetrics, box_iou=box_iou, init_seeds=init_seeds).items():
        setattr(U, k, v)


def make_targets(batch, S, C, device):
    """Ultralytics batch -> (B,S,S,5+C) grid targets, same convention as the original data_to_yolo.py
    (cell offset of the centre, w/h relative to the image, objectness, one-hot class; one box per cell, the last wins)."""
    B = batch["img"].shape[0]
    tgt = np.zeros((B, S, S, 5 + C), dtype=np.float32)
    bi = batch["batch_idx"].cpu().numpy().astype(int)
    cls = batch["cls"].cpu().numpy().reshape(-1).astype(int)
    bb = batch["bboxes"].cpu().numpy()          # normalised xywh in the letterboxed (square) image
    for k in range(len(bi)):
        cx, cy, w, h = bb[k]
        gi = min(max(int(cx * S), 0), S - 1)
        gj = min(max(int(cy * S), 0), S - 1)
        t = tgt[bi[k], gj, gi]
        t[0], t[1], t[2], t[3] = cx * S - gi, cy * S - gj, w, h
        t[4] = 1.0
        t[5 + cls[k]] = 1.0
    return torch.from_numpy(tgt).to(device)


def decode_and_nms(preds, S, imgsz, conf=0.001, iou=0.7, max_det=300):
    """preds (B,S,S,5+C) -> list of (n,6) [x1,y1,x2,y2,conf,cls] in pixels of the imgsz x imgsz image.
    score = sigmoid(objectness) * sigmoid(class logit); NMS exactly as the Ultralytics validator."""
    B, dev = preds.shape[0], preds.device
    gy, gx = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
    cx = (gx + preds[..., 0]) / S
    cy = (gy + preds[..., 1]) / S
    boxes = torch.stack([cx, cy, preds[..., 2], preds[..., 3]], -1) * imgsz
    scores = torch.sigmoid(preds[..., 4:5]) * torch.sigmoid(preds[..., 5:])
    pred = torch.cat([boxes, scores], -1).reshape(B, S * S, -1).transpose(1, 2)   # (B, 4+C, S*S)
    return U.ops.non_max_suppression(pred, conf_thres=conf, iou_thres=iou, max_det=max_det)


def match(det, cls, gt_xyxy, iouv):
    """Same matching as the Ultralytics DetectionValidator: (n_det, 10) bool, one column per IoU 0.50..0.95."""
    correct = np.zeros((det.shape[0], iouv.shape[0]), dtype=bool)
    iou = U.box_iou(gt_xyxy, det[:, :4])
    same = cls[:, None] == det[:, 5]
    for i in range(iouv.shape[0]):
        x = torch.where((iou >= iouv[i]) & same)
        if x[0].shape[0]:
            m = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                m = m[m[:, 2].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool)


@torch.no_grad()
def evaluate(model, loader, criterion, S, imgsz, names, device):
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats, lb, lc, n = [], 0.0, 0.0, 0
    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True).float() / 255
        preds = model(imgs)
        _, b, c = criterion(preds, make_targets(batch, S, len(names), device))
        lb += float(b.item()); lc += float(c.item()); n += 1
        for si, det in enumerate(decode_and_nms(preds, S, imgsz)):
            idx = batch["batch_idx"] == si
            cls = batch["cls"][idx].to(device).view(-1)
            gt = U.ops.xywh2xyxy(batch["bboxes"][idx].to(device)) * imgsz
            if det.shape[0] == 0:
                if cls.numel():
                    stats.append((torch.zeros(0, 10, dtype=torch.bool), torch.zeros(0), torch.zeros(0), cls.cpu()))
                continue
            correct = match(det, cls, gt, iouv) if cls.numel() else torch.zeros(det.shape[0], 10, dtype=torch.bool)
            stats.append((correct, det[:, 4].cpu(), det[:, 5].cpu(), cls.cpu()))
    metrics = U.DetMetrics(save_dir=Path("."), plot=False, names=names)
    out = {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0, "per_class": {}}
    if stats:
        s = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
        if len(s) and s[0].any():
            metrics.process(*s)
            out["precision"], out["recall"], out["map50"], out["map50_95"] = [float(v) for v in metrics.mean_results()]
            for pos, c in enumerate(metrics.box.ap_class_index):
                p, r, ap50, ap = metrics.box.class_result(pos)
                out["per_class"][str(names[int(c)])] = (float(p), float(r), float(ap50), float(ap))
    out["val_box_loss"], out["val_cls_loss"] = lb / max(n, 1), lc / max(n, 1)
    return out


def main():
    a = parse_args()
    a.project = os.path.abspath(a.project)
    run_dir = Path(a.project) / a.name
    wdir = run_dir / "weights"
    ckpt = None
    if a.resume:
        ckpt = torch.load(wdir / "last.pt", map_location="cpu", weights_only=False)
        keep = dict(resume=True, project=a.project, name=a.name)
        a = argparse.Namespace(**{**ckpt["args"], **keep})
        print(f"[resume] continuing from epoch {ckpt['epoch'] + 1}")
    elif (wdir / "best.pt").exists():
        raise SystemExit(f"{run_dir} already has results; use another --name")
    wdir.mkdir(parents=True, exist_ok=True)

    import_ultralytics(a.ultralytics_dir)
    U.init_seeds(a.seed)
    device = torch.device(f"cuda:{a.device}" if torch.cuda.is_available() else "cpu")
    data = U.check_det_dataset(a.data)
    names = data["names"]
    C = len(names)

    cfg = U.get_cfg(U.DEFAULT_CFG, dict(imgsz=a.imgsz, mosaic=a.mosaic, close_mosaic=a.close_mosaic, cache=a.cache,
                                        fraction=a.fraction, workers=a.workers, batch=a.batch, rect=False))
    train_ds = U.build_yolo_dataset(cfg, data["train"], a.batch, data, mode="train", rect=False, stride=32)
    val_ds = U.build_yolo_dataset(cfg, data["val"], a.batch, data, mode="val", rect=False, stride=32)
    train_loader = U.build_dataloader(train_ds, a.batch, a.workers, shuffle=True, rank=-1)
    val_loader = U.build_dataloader(val_ds, a.batch, a.workers, shuffle=False, rank=-1)

    model = SimpleYoloCNN(num_classes=C, img_size=a.imgsz, S=a.S).to(device)   # ImageNet ResNet18 backbone (as before)
    criterion = SimpleYoloLoss()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    start, best_fit = 0, -1.0
    if ckpt:
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["opt"]); sched.load_state_dict(ckpt["sched"])
        start, best_fit = ckpt["epoch"] + 1, ckpt["best_fit"]
    else:
        cfg_out = dict(vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), torch=torch.__version__,
                       ultralytics=U.version, host=platform.node(), params=sum(p.numel() for p in model.parameters()),
                       gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        try:
            cfg_out["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        (run_dir / "run_config.json").write_text(json.dumps(cfg_out, indent=2))
        print(json.dumps(cfg_out, indent=2))

    csv_path = run_dir / "epoch_metrics.csv"
    mosaic_closed = False
    for epoch in range(start, a.epochs):
        if a.mosaic > 0 and epoch >= a.epochs - a.close_mosaic and not mosaic_closed:
            train_ds.close_mosaic(hyp=cfg)
            if hasattr(train_loader, "reset"):
                train_loader.reset()
            mosaic_closed = True
            print(f"[epoch {epoch + 1}] mosaic closed")
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        model.train()
        sb = sc = 0.0
        nb = 0
        for batch in train_loader:
            imgs = batch["img"].to(device, non_blocking=True).float() / 255
            tgt = make_targets(batch, a.S, C, device)
            loss, box, cls = criterion(model(imgs), tgt)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sb += float(box.item()); sc += float(cls.item()); nb += 1
        train_seconds = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        sched.step()
        ev = evaluate(model, val_loader, criterion, a.S, a.imgsz, names, device)
        fit = 0.1 * ev["map50"] + 0.9 * ev["map50_95"]
        is_best = fit >= best_fit
        best_fit = max(best_fit, fit)
        epoch_seconds = time.time() - t0
        prev = 0.0
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            prev = float(rows[-1]["elapsed_seconds"]) if rows else 0.0
        nan = float("nan")
        row = {"epoch": epoch + 1, "train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
               "elapsed_seconds": prev + epoch_seconds,
               "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else nan,
               "lr_pg0": lr_now, "lr_pg1": nan, "lr_pg2": nan,
               "train_box_loss": sb / max(nb, 1), "val_box_loss": ev["val_box_loss"],
               "train_cls_loss": sc / max(nb, 1), "val_cls_loss": ev["val_cls_loss"],
               "train_dfl_loss": nan, "val_dfl_loss": nan,      # VanillaCNN has no DFL; its 'cls' loss includes objectness
               "precision": ev["precision"], "recall": ev["recall"], "map50": ev["map50"], "map50_95": ev["map50_95"],
               "fitness": fit, "is_best": int(is_best)}
        for n_ in names.values() if isinstance(names, dict) else names:
            p, r, ap50, ap = ev["per_class"].get(str(n_), (nan, nan, nan, nan))
            row[f"{n_}_p"], row[f"{n_}_r"], row[f"{n_}_ap50"], row[f"{n_}_ap50_95"] = p, r, ap50, ap
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"epoch {epoch + 1}/{a.epochs}  train {row['train_box_loss']:.3f}/{row['train_cls_loss']:.3f}  "
              f"mAP50 {ev['map50']:.4f}  mAP50-95 {ev['map50_95']:.4f}  P {ev['precision']:.3f}  R {ev['recall']:.3f}  "
              f"{epoch_seconds:.0f}s", flush=True)
        state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                 "epoch": epoch, "best_fit": best_fit, "args": vars(a)}
        torch.save(state, wdir / "last.pt")
        if is_best:
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "args": vars(a)}, wdir / "best.pt")
    print("done")


if __name__ == "__main__":
    main()