#!/usr/bin/env python3
"""Re-evaluates every model on two subsets of the test set, without running any network (CPU, seconds):

    unique      test images that have NO confirmed near-duplicate in train/val  (leakage_test_confirmed.txt, from check_leakage.py)
    duplicated  test images that DO have one
    all         the full test set (for reference; identical to the main result)

The operating threshold is the one chosen on validation for the main evaluation (metrics_seed0.json), so nothing is
re-tuned. Bootstrap confidence intervals resample images inside each subset.

    python eval_subset.py --confirmed ../Results_test/leakage/leakage_test_confirmed.txt
Output: <results>/summary/subset_metrics.csv   and a table printed on screen.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import metrics as MT

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "Results_test").exists() else HERE.parent
KEYS = ["map50", "map50_95", "precision", "recall", "f1", "ap50_smoke", "ap50_fire", "false_alarm_rate_empty_images"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "Results_test"))
    ap.add_argument("--confirmed", default=str(ROOT / "Results_test" / "leakage" / "leakage_test_confirmed.txt"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--name", default="subset_metrics", help="output file name (without .csv); use different names for different duplicate lists")
    ap.add_argument("--seed", type=int, default=0, help="which Poisson generator seed to use (0 = the one the threshold came from)")
    a = ap.parse_args()

    res = Path(a.results)
    dup_names = {Path(l.strip()).name for l in Path(a.confirmed).read_text().splitlines() if l.strip()}
    rows, out = [], []
    for d in sorted(p for p in res.iterdir() if p.is_dir() and (p / f"test_seed{a.seed}.npz").exists()):
        mj = d / f"metrics_seed0.json"
        if not mj.exists():
            continue
        thr = json.loads(mj.read_text())["threshold"]
        dump = MT.Dump(d / f"test_seed{a.seed}.npz")
        ev = MT.Eval(dump, thr)
        is_dup = np.array([Path(str(f)).name in dup_names for f in dump.files])
        subsets = {"all": np.ones(dump.n, bool), "unique": ~is_dup, "duplicated": is_dup}
        rng = np.random.default_rng(0)
        for name, mask in subsets.items():
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            point = ev.metrics(mask.astype(float))
            boots = {k: [] for k in KEYS}
            for _ in range(a.n_boot):
                w = np.bincount(rng.choice(idx, len(idx)), minlength=dump.n).astype(float)
                m = ev.metrics(w)
                for k in KEYS:
                    boots[k].append(m[k])
            row = dict(model=d.name, subset=name, n_images=int(len(idx)), threshold=thr)
            for k in KEYS:
                lo, hi = MT.ci(np.asarray(boots[k]))
                row[k], row[k + "_lo"], row[k + "_hi"] = point[k], lo, hi
            rows.append(row)
            out.append(f"{d.name:28s} {name:10s} n={len(idx):5d}  mAP50 {point['map50']:.3f} [{row['map50_lo']:.3f},{row['map50_hi']:.3f}]  "
                       f"mAP50-95 {point['map50_95']:.3f} [{row['map50_95_lo']:.3f},{row['map50_95_hi']:.3f}]  F1 {point['f1']:.3f}")
    if not rows:
        raise SystemExit("no models with test dumps and metrics_seed0.json found; run the accuracy stage first")
    import csv
    (res / "summary").mkdir(exist_ok=True)
    with open(res / "summary" / f"{a.name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"test images with a confirmed near-duplicate in train/val: {len(dup_names)} listed")
    print("\n".join(out))
    print(f"\nsaved {res / 'summary' / f'{a.name}.csv'}")


if __name__ == "__main__":
    main()