#!/usr/bin/env python3
"""Pipeline check for VanillaCNN (goes in VanillaCNN/, next to train_vanillacnn.py).
Feeds the *ground-truth grid targets* through the exact decode/NMS/metric code as if they were network outputs.
If the evaluation is correct, mAP50 must be close to 1 (only lost where two boxes fall in one cell).
A low value = bug in targets/decoding/evaluation, not a model that fails to learn.
    python oracle_check_vanilla.py --data ../datasets/overfit_subset/data.yaml
"""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_vanillacnn as T

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--S", type=int, default=20)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--split", default="val", choices=["val", "train"])
ap.add_argument("--ultralytics-dir", default="../SpikeYOLO")
a = ap.parse_args()

T.import_ultralytics(a.ultralytics_dir)
U = T.U
data = U.check_det_dataset(a.data)
names, C = data["names"], len(data["names"])
cfg = U.get_cfg(U.DEFAULT_CFG, dict(imgsz=a.imgsz, mosaic=0.0, cache="ram", workers=4, batch=16, rect=False))
ds = U.build_yolo_dataset(cfg, data[a.split], 16, data, mode="val", rect=False, stride=32)
loader = U.build_dataloader(ds, 16, 4, shuffle=False, rank=-1)

class Oracle(torch.nn.Module):
    """returns the logits that encode the targets of the batch it is told about"""
    def __init__(s): super().__init__(); s.t = None
    def forward(s, x):
        t = s.t.clone()
        out = t.clone()
        out[..., 4] = (t[..., 4] * 2 - 1) * 12            # objectness logit +-12
        out[..., 5:] = (t[..., 5:] * 2 - 1) * 12          # class logits +-12
        return out

class Crit:                                                # loss not needed for this check
    def __call__(s, p, t):
        z = torch.zeros((), device=p.device); return z, z, z

orc = Oracle()
class Wrap:
    def __init__(s, loader): s.loader = loader
    def __iter__(s):
        for b in s.loader:
            orc.t = T.make_targets(b, a.S, C, "cpu")
            yield b
dev = torch.device("cpu")
class M(torch.nn.Module):
    def eval(s): return s
    def __call__(s, x): return orc(x)
res = T.evaluate(M(), Wrap(loader), Crit(), a.S, a.imgsz, names, dev)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "per_class"})
print("OK: evaluation pipeline is consistent" if res["map50"] > 0.8 else "PROBLEM: oracle mAP50 is low -> targets/decoding/evaluation bug")