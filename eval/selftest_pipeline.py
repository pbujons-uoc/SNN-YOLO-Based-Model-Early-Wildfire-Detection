#!/usr/bin/env python3
"""End-to-end self-test of metrics + aggregate on synthetic dumps and fake benchmark files (numpy/pandas/scipy only):
    python selftest_pipeline.py"""
import json, tempfile
from pathlib import Path
import numpy as np
import aggregate as AG
import metrics as MT
from selftest_metrics import synth, write_dump, dets_from_gt

tmp = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
base = synth(300, rng, 0.8)
for name, q in [("spikeyolo_rgb_T4_s0", 0.85), ("spikeyolo_poisson_T4_s0", 0.6), ("spikeyolo_latency_T4_s0", 0.7), ("yolov8m_s0", 0.9)]:
    d = tmp / name; d.mkdir()
    seeds = 3 if "poisson" in name else 1
    for split in ("val", "test"):
        for s in range(seeds):
            n = 300 if split == "test" else 150
            gts = [g for _, g in synth(n, np.random.default_rng(5 if split == "test" else 6), 1.0)]   # same GT for every model
            r_ = np.random.default_rng(100 * s + (7 if split == "test" else 8) + int(q * 10))
            imgs = [(dets_from_gt(g, r_, q), g) for g in gts]
            write_dump(d / f"{split}_seed{s}.npz", imgs)
    thr = None
    for s in range(seeds):
        r = MT.evaluate(d / "val_seed0.npz", d / f"test_seed{s}.npz", d / f"metrics_seed{s}.json", thr, 100 if s == 0 else 0)
        thr = r["threshold"]
    for rep in range(1, 6):
        w = lambda lat, e: dict(n_iters=1000, batch=1, seconds=30, latency_ms_mean=lat + rng.normal(0, .1), latency_ms_median=lat,
                                latency_ms_p95=lat * 1.2, throughput_img_s=1000 / lat, power_mean_W=150 + rng.normal(0, 3),
                                energy_gross_mJ_per_image=e + 400, energy_net_mJ_per_image=e + rng.normal(0, 5))
        lat = 20 if "yolo" in name and "spike" not in name else 30
        b = dict(model=name, repeat=rep, batch=1, precision="fp32", params=23e6, windows=dict(network=w(lat, 3000), endtoend=w(lat + 3, 3200)))
        (d / f"bench_fp32_b1_r{rep}.json").write_text(json.dumps(b))
data = AG.load(tmp)
acc = AG.accuracy_table(data); eff = AG.efficiency_table(data, carbon=200.0); tests = AG.efficiency_tests(data)
pairs = AG.accuracy_pairs(tmp, data, 50)
assert len(acc) == 4 and len(eff) == 8 and len(tests) > 0 and len(pairs) > 0
assert acc.loc[acc.model == "spikeyolo_poisson_T4_s0", "n_seeds"].iloc[0] == 3
assert "map50_sd_seeds" in acc.columns and "map50_ci_lo" in acc.columns and "co2_mg_per_image_gross" in eff.columns
AG.write_tex(acc, eff, tmp / "t.tex")
print(acc[["label", "n_seeds", "threshold", "map50", "map50_ci_lo", "map50_ci_hi", "map50_sd_seeds"]].round(3).to_string(index=False))
print(eff[["label", "window", "latency_ms_mean", "latency_ms_mean_ci95", "energy_net_mJ_per_image"]].round(2).head(4).to_string(index=False))
print(tests[["window", "stat", "a", "b", "welch_p", "welch_p_holm", "mannwhitney_p"]].round(4).head(3).to_string(index=False))
print(pairs.round(3).head(3).to_string(index=False))
print("pipeline self-test passed")
