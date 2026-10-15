"""Model registry and loaders shared by dump_predictions.py and benchmark_efficiency.py.

Layout expected (the folder of this file is code_paper_2/eval):
    code_paper_2/Results_train/<run>/weights/best.pt     trained runs (queue)
    code_paper_2/SpikeYOLO        ultralytics 8.0.197 fork (SpikeYOLO and the VanillaCNN data pipeline)
    code_paper_2/ultralytics      ultralytics 8.3.231 (YOLOv8m)
    code_paper_2/VanillaCNN       model.py, loss.py, train_vanillacnn.py
    code_paper_2/datasets/D-Fire_clean/data.yaml

Every model family runs in its own process (two different ultralytics versions cannot share one interpreter).
All models are evaluated with the same data pipeline: square 640 x 640 letterbox, no augmentation, the same NMS
(conf 0.001, IoU 0.7, max 300 detections, multi-label as in the Ultralytics validator).
"""
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_TRAIN = ROOT / "Results_train"
RESULTS_TEST = ROOT / "Results_test"
DATA_YAML = ROOT / "datasets" / "D-Fire_clean" / "data.yaml"
FORK_DIR = ROOT / "SpikeYOLO"
YOLO_DIR = ROOT / "ultralytics"
VANILLA_DIR = ROOT / "VanillaCNN"
IMGSZ = 640


@dataclass
class Spec:
    name: str
    family: str                 # spikeyolo | yolov8 | vanilla
    weights: Path
    T: int = None
    encoding: str = None        # rgb | poisson | latency (spikeyolo only)

    @property
    def stochastic(self):       # only Poisson input depends on a random generator
        return self.encoding == "poisson"


_SPIKE = re.compile(r"^spikeyolo_(rgb|poisson|latency)_T(\d+)_s(\d+)(.*)$")


def discover(results_train=RESULTS_TRAIN):
    """All finished runs that have a best.pt (quick/probe runs and folders starting with '_' are ignored)."""
    specs = []
    for d in sorted(Path(results_train).iterdir()):
        w = d / "weights" / "best.pt"
        if not d.is_dir() or d.name.startswith("_") or "quick" in d.name or "probe" in d.name or not w.exists():
            continue
        m = _SPIKE.match(d.name)
        if m:
            specs.append(Spec(d.name, "spikeyolo", w, int(m.group(2)), m.group(1)))
        elif d.name.startswith("yolov8"):
            specs.append(Spec(d.name, "yolov8", w))
        elif d.name.startswith("vanillacnn"):
            specs.append(Spec(d.name, "vanilla", w))
    return specs


def get_spec(name, results_train=RESULTS_TRAIN):
    for s in discover(results_train):
        if s.name == name:
            return s
    raise SystemExit(f"no finished run named {name!r} with weights/best.pt in {results_train}")


# ------------------------------------------------------------------------------------------ ultralytics
def trust_checkpoints():
    """torch >= 2.6 defaults to weights_only=True, which cannot unpickle our own Ultralytics checkpoints."""
    orig = torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)
    torch.load = _load


def import_ultralytics(path):
    """Import the ultralytics package that lives in `path` and return the pieces we use."""
    path = str(Path(path).resolve())
    if "ultralytics" in sys.modules:
        have = str(Path(sys.modules["ultralytics"].__file__).resolve().parents[1])
        if have != path:
            raise RuntimeError(f"ultralytics already imported from {have}, wanted {path}")
    else:
        sys.path.insert(0, path)
    import ultralytics
    assert str(Path(ultralytics.__file__).resolve()).startswith(path), f"wrong ultralytics: {ultralytics.__file__}"
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, ops
    return SimpleNamespace(version=ultralytics.__version__, file=ultralytics.__file__, get_cfg=get_cfg,
                           build_dataloader=build_dataloader, build_yolo_dataset=build_yolo_dataset,
                           check_det_dataset=check_det_dataset, DEFAULT_CFG=DEFAULT_CFG, ops=ops)


def build_loader(U, split, batch, workers=4, cache="ram", data_yaml=DATA_YAML):
    """Ultralytics dataset in 'val' mode (letterbox, no augmentation), square 640 x 640, fixed order."""
    data = U.check_det_dataset(str(data_yaml))
    img_path = data.get(split) or str(Path(data["path"]) / split)    # data.yaml may have no 'test:' key
    if not Path(img_path).exists():
        raise SystemExit(f"split {split!r} not found at {img_path}; add '{split}: {split}/' to {data_yaml}")
    cfg = U.get_cfg(U.DEFAULT_CFG, dict(imgsz=IMGSZ, mosaic=0.0, cache=cache, workers=workers, batch=batch, rect=False))
    ds = U.build_yolo_dataset(cfg, img_path, batch, data, mode="val", rect=False, stride=32)
    loader = U.build_dataloader(ds, batch, workers, shuffle=False, rank=-1)
    return loader, data["names"]


# ------------------------------------------------------------------------------------------ models
class Loaded:
    """net(x) takes float images [B,3,640,640] in [0,1]; decode(out) -> list of (n,6) [x1,y1,x2,y2,conf,cls]."""

    def __init__(self, spec, net, decode, U, layer0=None, extra=None):
        self.spec, self.net, self.decode, self.U, self.layer0 = spec, net, decode, U, layer0
        self.params = sum(p.numel() for p in net.parameters())
        self.extra = extra or {}


def get_nms(U):
    """non_max_suppression lives in utils.ops in Ultralytics 8.0.x (SpikeYOLO fork) and in utils.nms in recent 8.3.x."""
    if hasattr(U.ops, "non_max_suppression"):
        return U.ops.non_max_suppression
    from ultralytics.utils import nms as _nms
    return _nms.non_max_suppression


def _nms_decode(U, multi_label):
    nms = get_nms(U)

    def decode(out, conf=0.001, iou=0.7, max_det=300):
        pred = out[0] if isinstance(out, (list, tuple)) else out
        return nms(pred.float(), conf, iou, multi_label=multi_label, max_det=max_det)
    return decode


def load(spec, device, multi_label=True):
    if spec.family in ("spikeyolo", "yolov8"):
        U = import_ultralytics(FORK_DIR if spec.family == "spikeyolo" else YOLO_DIR)
        trust_checkpoints()
        from ultralytics import YOLO
        net = YOLO(str(spec.weights)).model.to(device).float().eval()
        for p in net.parameters():
            p.requires_grad_(False)
        layer0 = next((m for m in net.modules() if m.__class__.__name__ == "MS_GetT"), None)
        if spec.family == "spikeyolo":
            assert layer0 is not None, "SpikeYOLO checkpoint without MS_GetT layer"
            assert getattr(layer0, "mode", "repeat") == (("repeat" if spec.encoding == "rgb" else spec.encoding)), (
                f"{spec.name}: checkpoint layer-0 mode is {getattr(layer0, 'mode', None)!r}, expected {spec.encoding!r}")
            assert layer0.T == spec.T, f"{spec.name}: checkpoint T={layer0.T}, expected {spec.T}"
        return Loaded(spec, net, _nms_decode(U, multi_label), U, layer0)

    if spec.family == "vanilla":
        sys.path.insert(0, str(VANILLA_DIR))
        import train_vanillacnn as TV
        TV.import_ultralytics(str(FORK_DIR))
        U = TV.U
        trust_checkpoints()
        ck = torch.load(spec.weights, map_location="cpu")
        a = ck["args"]
        data = U.check_det_dataset(str(DATA_YAML))
        net = TV.SimpleYoloCNN(num_classes=len(data["names"]), img_size=a["imgsz"], S=a["S"])
        net.load_state_dict(ck["model"])
        net = net.to(device).float().eval()
        for p in net.parameters():
            p.requires_grad_(False)
        S, imgsz = a["S"], a["imgsz"]
        nms = get_nms(U)

        def decode(preds, conf=0.001, iou=0.7, max_det=300):
            B, dev = preds.shape[0], preds.device
            gy, gx = torch.meshgrid(torch.arange(S, device=dev), torch.arange(S, device=dev), indexing="ij")
            boxes = torch.stack([(gx + preds[..., 0]) / S, (gy + preds[..., 1]) / S, preds[..., 2], preds[..., 3]], -1) * imgsz
            scores = torch.sigmoid(preds[..., 4:5]) * torch.sigmoid(preds[..., 5:])
            pred = torch.cat([boxes, scores], -1).reshape(B, S * S, -1).transpose(1, 2)
            return nms(pred.float(), conf, iou, multi_label=multi_label, max_det=max_det)
        return Loaded(spec, net, decode, U, None, {"S": S})
    raise ValueError(spec.family)


def device_from(arg):
    return torch.device(f"cuda:{arg}" if str(arg).isdigit() and torch.cuda.is_available() else str(arg))