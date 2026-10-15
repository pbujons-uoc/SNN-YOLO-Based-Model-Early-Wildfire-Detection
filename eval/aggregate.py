#!/usr/bin/env python3
"""Collect every metrics_seed*.json and bench_*.json under Results_test and write the summary tables.

    python aggregate.py --results ../Results_test [--carbon-intensity 200]   # g CO2 per kWh of the grid you assume

Writes into <results>/_summary/:
  accuracy.csv        per model: mAP50, mAP50-95, per-class AP, P/R/F1, TP/FP/FN, threshold, recall by size,
                      with the 95 % bootstrap interval over test images (and mean +- sd over generator seeds for Poisson)
  accuracy_pairs.csv  paired bootstrap difference between selected pairs of models (same images, same resamples)
  efficiency.csv      per model and measurement window: mean, sd and 95 % CI over the repeats
  efficiency_tests.csv  Welch t-test and Mann-Whitney U between models, Holm-corrected, with Cohen's d
  tables.tex          the two main tables, ready to paste
The repeats of a deterministic model give identical accuracy, so its accuracy uncertainty is the bootstrap interval,
not a spread over repeats. Poisson is the exception: its spread over generator seeds is reported as well.
Variance caused by training (different training seeds) is NOT measured here: only seed 0 was trained.
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import metrics as MT

ACC_KEYS = ["map50", "map50_95", "ap50_smoke", "ap50_fire", "ap50_95_smoke", "ap50_95_fire", "precision", "recall", "f1",
            "f1_macro", "tp", "fp", "fn", "false_alarm_rate_empty_images"]
EFF_STATS = ["latency_ms_mean", "latency_ms_median", "latency_ms_p95", "throughput_img_s", "power_mean_W",
             "energy_gross_mJ_per_image", "energy_net_mJ_per_image", "cc_energy_gross_mJ_per_image", "cc_energy_net_mJ_per_image",
             "cc_co2_mg_per_image_gross", "cc_co2_mg_per_image_net"]
WINDOWS = ["network", "encode", "endtoend"]


def pretty(name):
    m = re.match(r"^spikeyolo_(rgb|poisson|latency)_T(\d+)_s\d+(.*)$", name)
    if m:
        enc = {"rgb": "direct RGB", "poisson": "Poisson", "latency": "latency"}[m.group(1)]
        return f"SpikeYOLO {enc}, T={m.group(2)}{m.group(3).replace('_', ' ')}"
    if name.startswith("yolov8m"):
        return "YOLOv8m"
    if name.startswith("vanillacnn_long"):
        return "VanillaCNN (180 epochs)"
    if name.startswith("vanillacnn"):
        return "VanillaCNN"
    return name


def load(results):
    out = {}
    for d in sorted(Path(results).iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        acc = [json.loads(p.read_text()) for p in sorted(d.glob("metrics_seed*.json"))]
        ben = [json.loads(p.read_text()) for p in sorted(d.glob("bench_*.json")) if not p.name.endswith("latencies.json")]
        if acc or ben:
            out[d.name] = dict(acc=acc, bench=ben)
    return out


def accuracy_table(data):
    rows = []
    for name, v in data.items():
        if not v["acc"]:
            continue
        a0 = v["acc"][0]
        r = {"model": name, "label": pretty(name), "n_seeds": len(v["acc"]), "threshold": a0["threshold"],
             "n_test_images": a0["n_test_images"]}
        for k in ACC_KEYS:
            vals = np.array([a["point"][k] for a in v["acc"]], float)
            r[k] = float(vals.mean())
            if len(vals) > 1:
                r[k + "_sd_seeds"] = float(vals.std(ddof=1))
            if "bootstrap95" in a0 and k in a0["bootstrap95"]:
                r[k + "_ci_lo"], r[k + "_ci_hi"] = a0["bootstrap95"][k]
        for k, val in a0["point"].items():
            if k.startswith(("recall_smoke_size", "recall_fire_size", "n_gt_")) or k.startswith(("precision_", "recall_", "f1_")):
                r.setdefault(k, val)
        rows.append(r)
    return pd.DataFrame(rows)


def efficiency_table(data, ci_level=0.95, carbon=None):
    rows = []
    for name, v in data.items():
        for w in WINDOWS:
            reps = [b["windows"][w] for b in v["bench"] if w in b["windows"]]
            if not reps:
                continue
            r = {"model": name, "label": pretty(name), "window": w, "n_repeats": len(reps),
                 "batch": v["bench"][0]["batch"], "precision": v["bench"][0]["precision"], "params": v["bench"][0]["params"]}
            for s in EFF_STATS:
                x = np.array([q[s] for q in reps if q.get(s) is not None], float)
                if not len(x):
                    continue
                r[s] = float(x.mean())
                if len(x) > 1:
                    r[s + "_sd"] = float(x.std(ddof=1))
                    r[s + "_ci95"] = float(stats.t.ppf(0.5 + ci_level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
            if carbon and "energy_gross_mJ_per_image" in r:
                r["co2_mg_per_image_gross"] = r["energy_gross_mJ_per_image"] / 3.6e9 * carbon * 1000.0
            rows.append(r)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def efficiency_tests(data, stats_=("latency_ms_mean", "energy_net_mJ_per_image")):
    rows = []
    for w in WINDOWS:
        for s in stats_:
            samples = {n: np.array([b["windows"][w][s] for b in v["bench"] if w in b["windows"] and b["windows"][w].get(s) is not None], float)
                       for n, v in data.items()}
            samples = {n: x for n, x in samples.items() if len(x) >= 2}
            block = []
            for a, b in itertools.combinations(sorted(samples), 2):
                x, y = samples[a], samples[b]
                t = stats.ttest_ind(x, y, equal_var=False)
                try:
                    u = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                except ValueError:
                    u = float("nan")
                sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
                block.append(dict(window=w, stat=s, a=a, b=b, mean_a=x.mean(), mean_b=y.mean(), n_a=len(x), n_b=len(y),
                                  diff=x.mean() - y.mean(), welch_p=float(t.pvalue), mannwhitney_p=float(u),
                                  cohens_d=float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")))
            if block:
                adj = holm([r["welch_p"] for r in block])
                for r, p in zip(block, adj):
                    r["welch_p_holm"] = float(p)
                rows += block
    return pd.DataFrame(rows)


def default_pairs(names):
    pairs = []
    sp = {}
    for n in names:
        m = re.match(r"^spikeyolo_(rgb|poisson|latency)_T(\d+)_s0$", n)
        if m:
            sp[(m.group(1), int(m.group(2)))] = n
    for T in sorted({t for _, t in sp}):
        pairs += [(sp[(a, T)], sp[(b, T)]) for a, b in itertools.combinations(["rgb", "poisson", "latency"], 2) if (a, T) in sp and (b, T) in sp]
    for e in ["rgb", "poisson", "latency"]:
        if (e, 4) in sp and (e, 6) in sp:
            pairs.append((sp[(e, 4)], sp[(e, 6)]))
    y = next((n for n in names if n.startswith("yolov8m")), None)
    if y:
        pairs += [(sp[k], y) for k in sorted(sp) if k[1] == 4]
    return pairs


def accuracy_pairs(results, data, n_boot, pairs=None):
    rows = []
    names = [n for n, v in data.items() if v["acc"]]
    for a, b in (pairs or default_pairs(names)):
        if a not in names or b not in names:
            continue
        try:
            ea = _eval(results, a, data[a]); eb = _eval(results, b, data[b])
            res = MT.paired_difference(ea, eb, n_boot=n_boot)
        except Exception as e:                              # e.g. different image order
            print(f"[pair {a} vs {b}] skipped: {e}")
            continue
        for k, v in res.items():
            rows.append(dict(a=a, b=b, metric=k, diff=v["diff"], ci95_lo=v["ci95"][0], ci95_hi=v["ci95"][1], p_boot=v["p_two_sided"]))
    return pd.DataFrame(rows)


def _eval(results, name, v):
    a0 = v["acc"][0]
    return MT.Eval(MT.Dump(Path(results) / name / "test_seed0.npz"), a0["threshold"], a0["names"])


def fmt(mean, sd=None, ci=None, nd=3):
    s = f"{mean:.{nd}f}"
    if sd is not None and not np.isnan(sd):
        s += f" $\\pm$ {sd:.{nd}f}"
    return s


def write_tex(acc, eff, path):
    L = []
    if len(acc):
        L += ["% accuracy (test set); brackets = 95% bootstrap interval over test images", "\\begin{tabular}{lcccccc}", "\\hline",
              "Model & mAP50 & mAP50-95 & Precision & Recall & F1 & Thr. \\\\", "\\hline"]
        for _, r in acc.iterrows():
            def c(k):
                lo, hi = r.get(k + "_ci_lo"), r.get(k + "_ci_hi")
                return f"{r[k]:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None and not pd.isna(lo) else "")
            L.append(f"{r['label']} & {c('map50')} & {c('map50_95')} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['threshold']:.2f} \\\\")
        L += ["\\hline", "\\end{tabular}", ""]
    if len(eff):
        L += ["% efficiency; mean $\\pm$ sd over repeats", "\\begin{tabular}{llcccc}", "\\hline",
              "Model & Window & Latency (ms) & Net energy (mJ/img) & Power (W) & CO$_2$ (mg/img) \\\\", "\\hline"]
        for _, r in eff.iterrows():
            L.append(f"{r['label']} & {r['window']} & {fmt(r['latency_ms_mean'], r.get('latency_ms_mean_sd'), nd=2)} & "
                     f"{fmt(r.get('energy_net_mJ_per_image', np.nan), r.get('energy_net_mJ_per_image_sd'), nd=1)} & "
                     f"{fmt(r.get('power_mean_W', np.nan), r.get('power_mean_W_sd'), nd=0)} & "
                     f"{fmt(r.get('cc_co2_mg_per_image_gross', np.nan), r.get('cc_co2_mg_per_image_gross_sd'), nd=4)} \\\\")
        L += ["\\hline", "\\end{tabular}"]
    Path(path).write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "Results_test"))
    ap.add_argument("--carbon-intensity", type=float, default=None, help="g CO2 per kWh of the grid (adds a CO2 column)")
    ap.add_argument("--n-boot-pairs", type=int, default=1000)
    ap.add_argument("--no-pairs", action="store_true")
    a = ap.parse_args()
    out = Path(a.results) / "_summary"
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.results)
    acc, eff = accuracy_table(data), efficiency_table(data, carbon=a.carbon_intensity)
    tests = efficiency_tests(data)
    acc.to_csv(out / "accuracy.csv", index=False)
    eff.to_csv(out / "efficiency.csv", index=False)
    tests.to_csv(out / "efficiency_tests.csv", index=False)
    if not a.no_pairs:
        accuracy_pairs(a.results, data, a.n_boot_pairs).to_csv(out / "accuracy_pairs.csv", index=False)
    write_tex(acc, eff, out / "tables.tex")
    pd.set_option("display.width", 200)
    if len(acc):
        print(acc[["label", "n_seeds", "threshold", "map50", "map50_95", "precision", "recall", "f1"]].round(4).to_string(index=False))
    if len(eff):
        cols = [c for c in ["label", "window", "n_repeats", "latency_ms_mean", "energy_net_mJ_per_image", "power_mean_W"] if c in eff]
        print(eff[cols].round(3).to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
