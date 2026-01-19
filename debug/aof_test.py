# inspect_pt.py
import torch
from pathlib import Path

def describe(x, prefix=""):
    if hasattr(x, "shape"):
        print(f"{prefix}Tensor: dtype={x.dtype}, shape={tuple(x.shape)}, min={x.min().item() if x.numel() else 'n/a'}, max={x.max().item() if x.numel() else 'n/a'}")
    else:
        print(f"{prefix}{type(x)}")

def main():
    pt_path = Path("./AoF00000.pt")
    obj = torch.load(pt_path, map_location="cpu")

    print("Loaded:", pt_path)
    print("Top-level type:", type(obj))

    if isinstance(obj, dict):
        print("Dict keys:", list(obj.keys()))
        for k, v in obj.items():
            describe(v, prefix=f"  [{k}] ")
    elif isinstance(obj, (list, tuple)):
        print(f"{type(obj).__name__} length:", len(obj))
        for i, v in enumerate(obj):
            describe(v, prefix=f"  [{i}] ")
    else:
        describe(obj)

if __name__ == "__main__":
    main()
