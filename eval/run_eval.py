#!/usr/bin/env python3
"""Orchestrates the whole test-set evaluation. Resumable: every step is skipped if its output file exists.

    python run_eval.py --stage accuracy                 # dumps val+test predictions of every finished run, then metrics
    python run_eval.py --stage bench --repeats 5        # time/energy; refuses while a training is running
    python run_eval.py --stage aggregate [--carbon-intensity 200]
    python run_eval.py --stage all --repeats 5
    python run_eval.py --stage accuracy --models yolov8m_s0 spikeyolo_rgb_T4_s0       # only some runs

Accuracy does not depend on timing, so it can run while the training queue is still going (it only slows it down);
the benchmark must run on an otherwise idle machine, after the queue has finished.
Run it with nohup if the session may close:
    nohup setsid python run_eval.py --stage all --repeats 5 > eval_main.log 2>&1 &
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import models as M

HERE = Path(__file__).resolve().parent


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE)
    if r.returncode != 0:
        print(f"!! failed ({r.returncode}): {' '.join(map(str, cmd))}", flush=True)
    return r.returncode == 0


def training_running():
    r = subprocess.run(["pgrep", "-f", "run_queue.py"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def stage_accuracy(specs, out, repeats, args):
    import metrics as MT
    for s in specs:
        seeds = repeats if s.stochastic else 1
        for split in ("val", "test"):
            run(["dump_predictions.py", "--model", s.name, "--split", split, "--seeds", seeds, "--out", out,
                 "--batch", args.dump_batch, "--precision", args.precision])
        d = Path(out) / s.name
        if not (d / "val_seed0.npz").exists():
            print(f"[{s.name}] no val dump, skipping metrics")
            continue
        thr = None
        for seed in range(seeds):
            tpath = d / f"test_seed{seed}.npz"
            opath = d / f"metrics_seed{seed}.json"
            if opath.exists() or not tpath.exists():
                continue
            r = MT.evaluate(d / "val_seed0.npz", tpath, opath, threshold=thr, n_boot=args.n_boot if seed == 0 else 0)
            thr = r["threshold"]                           # the threshold of seed 0 is reused for the other seeds
            p = r["point"]
            print(f"[{s.name} seed {seed}] thr {thr:.2f}  mAP50 {p['map50']:.4f}  mAP50-95 {p['map50_95']:.4f}  "
                  f"P {p['precision']:.3f} R {p['recall']:.3f} F1 {p['f1']:.3f}", flush=True)


def stage_bench(specs, out, repeats, args):
    if training_running() and not args.allow_busy:
        raise SystemExit("run_queue.py is still running: the benchmark needs an idle GPU. Wait, or use --allow-busy for a dry run.")
    for r in range(1, repeats + 1):
        order = list(specs)
        random.Random(r).shuffle(order)                    # interleave models; different order in every repeat
        for s in order:
            ok = run(["benchmark_efficiency.py", "--model", s.name, "--repeat", r, "--out", out, "--batch", args.bench_batch,
                      "--precision", args.precision, "--fuse", args.fuse] + (["--allow-busy"] if args.allow_busy else []))
            time.sleep(args.cooldown if ok else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["accuracy", "bench", "aggregate", "all"])
    ap.add_argument("--models", nargs="*", default=None, help="run names (default: every finished run in Results_train)")
    ap.add_argument("--repeats", type=int, default=5, help="benchmark repeats and Poisson generator seeds")
    ap.add_argument("--out", default=str(M.RESULTS_TEST))
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    ap.add_argument("--dump-batch", type=int, default=8)
    ap.add_argument("--bench-batch", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--cooldown", type=float, default=30.0, help="seconds between benchmark runs")
    ap.add_argument("--carbon-intensity", type=float, default=None)
    ap.add_argument("--fuse", default="auto", choices=["auto", "no"], help="Conv+BN fusion in the benchmark (use 'no' for the same code path in every model)")
    ap.add_argument("--allow-busy", action="store_true")
    a = ap.parse_args()

    specs = M.discover()
    if a.models:
        specs = [s for s in specs if s.name in a.models]
        missing = set(a.models) - {s.name for s in specs}
        if missing:
            raise SystemExit(f"not found / not finished: {sorted(missing)}")
    print("models:", ", ".join(s.name for s in specs), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.stage in ("accuracy", "all"):
        stage_accuracy(specs, a.out, a.repeats, a)
    if a.stage in ("bench", "all"):
        stage_bench(specs, a.out, a.repeats, a)
    if a.stage in ("aggregate", "all"):
        run(["aggregate.py", "--results", a.out] + (["--carbon-intensity", a.carbon_intensity] if a.carbon_intensity else []))


if __name__ == "__main__":
    main()