"""Reverse Poisson encoded .pt files to RGB images and overlay YOLO boxes.

Usage examples:
    python debug/un_encoding_poisson.py --pt datasets/poisson_4/val/images/AoF01029.pt --labels datasets/poisson_4/val/labels --out debug/recon

    python debug/un_encoding_poisson.py --pt datasets/poisson_4/val/images --labels datasets/poisson_4/val/labels --out debug/recon

Notes:
- Expects .pt files shaped [T, C, H, W] (dtype uint8 or float).
- Reconstruction uses the mean firing rate across time as intensity.
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np
import cv2

CLASS_NAMES = {0: 'smoke', 1: 'fire'}


def recon_poisson(spikes: torch.Tensor) -> np.ndarray:
    """Reconstruct image from Poisson-encoded spikes by averaging across T.
    spikes: [T, C, H, W] (0/1 uint8 or float)
    returns: H x W x C uint8 (RGB)
    """
    if spikes.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {spikes.shape}")
    s = spikes.float()
    prob = s.mean(dim=0)  # [C,H,W]
    prob = prob.clamp(0.0, 1.0)
    img = (prob * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return img


def read_yolo_labels(label_path: Path, img_h: int, img_w: int):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            boxes.append({'cls': cls, 'xyxy': [int(x1), int(y1), int(x2), int(y2)]})
    return boxes


def draw_boxes(img: np.ndarray, boxes, thickness=2):
    out = img.copy()
    for b in boxes:
        cls = b['cls']
        x1, y1, x2, y2 = b['xyxy']
        name = CLASS_NAMES.get(cls, str(cls))
        color = (0, 0, 255) if name == 'fire' else (0, 255, 255)  # BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, name, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def process_file(pt_path: Path, labels_dir: Path, out_dir: Path):
    try:
        spikes = torch.load(pt_path)
    except Exception as e:
        print(f"Failed to load {pt_path}: {e}")
        return
    if isinstance(spikes, torch.Tensor) is False:
        spikes = torch.tensor(spikes)
    if spikes.ndim == 5:
        spikes = spikes.squeeze(0)
    if spikes.ndim == 3:
        spikes = spikes.unsqueeze(0)
    T, C, H, W = spikes.shape
    img = recon_poisson(spikes)
    lbl_name = pt_path.stem + '.txt'
    lbl_path = labels_dir / lbl_name
    boxes = read_yolo_labels(lbl_path, H, W)
    out_img = draw_boxes(img, boxes)
    out_path = out_dir / (pt_path.stem + '_recon.png')
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"Saved reconstruction to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Un-encode poisson .pt spike files and overlay YOLO boxes')
    parser.add_argument('--pt', required=True, help='Path to .pt file or directory with .pt files')
    parser.add_argument('--labels', default=None, help='Directory with YOLO labels (if not provided, replaces images->labels)')
    parser.add_argument('--out', default='debug/recon_poisson', help='Output directory')
    args = parser.parse_args()

    pt_path = Path(args.pt)
    out_dir = Path(args.out)

    if pt_path.is_dir():
        pt_files = sorted(pt_path.glob('*.pt'))
    elif pt_path.is_file():
        pt_files = [pt_path]
    else:
        print(f"Path not found: {pt_path}")
        return

    if args.labels:
        labels_dir = Path(args.labels)
    else:
        if 'images' in str(pt_path):
            labels_dir = Path(str(pt_path).replace('images', 'labels'))
        else:
            labels_dir = pt_path.parent.parent / 'labels'

    for f in pt_files:
        process_file(f, labels_dir, out_dir)

    print('Done.')

if __name__ == '__main__':
    main()
