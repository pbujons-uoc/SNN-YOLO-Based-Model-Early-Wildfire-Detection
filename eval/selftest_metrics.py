#!/usr/bin/env python3
"""Self-test of metrics.py on synthetic data (numpy only, runs anywhere):  python selftest_metrics.py"""
import tempfile
from pathlib import Path

import numpy as np

import metrics as M


def write_dump(path, images):
    """images: list of (dets (n,6), gts (m,5))"""
    dets = [d for d, _ in images]; gts = [g for _, g in images]
    np.savez_compressed(
        path, files=np.array([f"img{i}" for i in range(len(images))]),
        det_ptr=np.concatenate([[0], np.cumsum([len(d) for d in dets])]).astype(np.int64),
        dets=np.concatenate(dets).astype(np.float32).reshape(-1, 6),
        gt_ptr=np.concatenate([[0], np.cumsum([len(g) for g in gts])]).astype(np.int64),
        gts=np.concatenate(gts).astype(np.float32).reshape(-1, 5))


def dets_from_gt(g, rng, quality):
    d = []
    for k in range(len(g)):
        if rng.random() < quality:
            j = rng.normal(0, 4, 4)
            d.append([g[k, 1] + j[0], g[k, 2] + j[1], g[k, 3] + j[2], g[k, 4] + j[3], rng.uniform(0.3, 1.0), g[k, 0]])
    for _ in range(rng.integers(0, 3)):
        x, y = rng.uniform(0, 500, 2); w, h = rng.uniform(8, 120, 2)
        d.append([x, y, x + w, y + h, rng.uniform(0.001, 0.6), rng.integers(0, 2)])
    d = np.array(d).reshape(-1, 6)
    return d[np.argsort(-d[:, 4])]


def synth(n_img, rng, quality=0.8):
    """random images with 0-3 GT boxes; detections = noisy copies of some GT (+ conf) plus random false positives"""
    out = []
    for _ in range(n_img):
        m = rng.integers(0, 4)
        g = np.zeros((m, 5))
        for k in range(m):
            x, y = rng.uniform(0, 500, 2); w, h = rng.uniform(8, 120, 2)
            g[k] = [rng.integers(0, 2), x, y, x + w, y + h]
        d = []
        for k in range(m):
            if rng.random() < quality:
                j = rng.normal(0, 4, 4)
                d.append([g[k, 1] + j[0], g[k, 2] + j[1], g[k, 3] + j[2], g[k, 4] + j[3], rng.uniform(0.3, 1.0), g[k, 0]])
        for _ in range(rng.integers(0, 3)):
            x, y = rng.uniform(0, 500, 2); w, h = rng.uniform(8, 120, 2)
            d.append([x, y, x + w, y + h, rng.uniform(0.001, 0.6), rng.integers(0, 2)])
        d = np.array(d).reshape(-1, 6)
        d = d[np.argsort(-d[:, 4])]
        out.append((d, g))
    return out


def reference_map(images, nc=2):
    """straightforward re-implementation of Ultralytics ap_per_class over concatenated stats (unit weights)"""
    tp, conf, pcls, tcls = [], [], [], []
    for d, g in images:
        c, _ = M.match(d, g, M.IOUV)
        tp.append(c); conf.append(d[:, 4]); pcls.append(d[:, 5]); tcls.append(g[:, 0])
    tp, conf, pcls, tcls = np.concatenate(tp), np.concatenate(conf), np.concatenate(pcls), np.concatenate(tcls)
    i = np.argsort(-conf); tp, conf, pcls = tp[i], conf[i], pcls[i]
    ap = np.zeros((nc, 10))
    for c in range(nc):
        n_l = (tcls == c).sum(); m = pcls == c
        if n_l == 0 or m.sum() == 0:
            continue
        fpc = (1 - tp[m]).cumsum(0); tpc = tp[m].cumsum(0)
        rec = tpc / (n_l + 1e-16); pre = tpc / (tpc + fpc)
        for k in range(10):
            ap[c, k] = M.compute_ap(rec[:, k], pre[:, k])
    return ap[:, 0].mean(), ap.mean()


def main():
    rng = np.random.default_rng(1)
    tmp = Path(tempfile.mkdtemp())

    # 1) perfect detections -> AP at the Ultralytics ceiling 0.995 (101-point interpolation), P = R = F1 = 1
    imgs = []
    for d, g in synth(60, rng):
        det = np.concatenate([g[:, 1:5], np.full((len(g), 1), 0.9), g[:, :1]], 1)
        imgs.append((det, g))
    write_dump(tmp / "perfect.npz", imgs)
    e = M.Eval(M.Dump(tmp / "perfect.npz"), 0.5)
    m = e.metrics()
    assert abs(m["map50"] - 0.995) < 1e-6 and abs(m["map50_95"] - 0.995) < 1e-6 and abs(m["f1"] - 1) < 1e-9, m
    print("1 perfect detections: OK")

    # 2) metrics.py == independent reference implementation
    imgs = synth(400, rng)
    write_dump(tmp / "rand.npz", imgs)
    dump = M.Dump(tmp / "rand.npz")
    ev = M.Eval(dump, 0.4)
    m = ev.metrics()
    r50, r5095 = reference_map(imgs)
    assert abs(m["map50"] - r50) < 1e-9 and abs(m["map50_95"] - r5095) < 1e-9, (m["map50"], r50, m["map50_95"], r5095)
    print(f"2 matches reference implementation: mAP50 {m['map50']:.4f}  mAP50-95 {m['map50_95']:.4f}: OK")

    # 3) bootstrap weights == physically duplicating the images
    idx = rng.integers(0, len(imgs), len(imgs))
    w = np.bincount(idx, minlength=len(imgs)).astype(float)
    dup = [imgs[i] for i in idx]
    write_dump(tmp / "dup.npz", dup)
    m_dup = M.Eval(M.Dump(tmp / "dup.npz"), 0.4).metrics()
    m_w = ev.metrics(w)
    for k in ["map50", "map50_95", "precision", "recall", "f1", "tp", "fp", "fn", "ap50_smoke", "ap50_fire",
              "false_alarm_rate_empty_images"]:
        assert abs(m_dup[k] - m_w[k]) < 1e-9, (k, m_dup[k], m_w[k])
    print("3 bootstrap multiplicities == duplicated images: OK")

    # 4) threshold selection + counts consistent
    thr, curve = M.select_threshold(dump)
    assert 0.01 <= thr <= 0.95 and max(curve["f1_mean_class"]) == curve["f1_mean_class"][int(round((thr - 0.01) / 0.01))]
    tot_gt = sum(len(g) for _, g in imgs)
    ev2 = M.Eval(dump, thr).metrics()
    assert abs(ev2["tp"] + ev2["fn"] - tot_gt) < 1e-9
    print(f"4 threshold {thr:.2f}: TP {ev2['tp']:.0f} + FN {ev2['fn']:.0f} = {tot_gt} GT: OK")

    # 5) full evaluate(), bootstrap interval brackets the point estimate, paired difference
    r = M.evaluate(tmp / "rand.npz", tmp / "rand.npz", tmp / "out.json", n_boot=200)
    lo, hi = r["bootstrap95"]["map50"]
    assert lo <= r["point"]["map50"] <= hi, (lo, r["point"]["map50"], hi)
    imgs_b = synth(400, np.random.default_rng(1), quality=0.4)     # worse model on images with the same ids
    write_dump(tmp / "worse.npz", imgs_b)
    pd = M.paired_difference(M.Eval(M.Dump(tmp / "rand.npz"), 0.4), M.Eval(M.Dump(tmp / "worse.npz"), 0.4), n_boot=200)
    print("5 evaluate + paired difference: OK", {k: round(v["diff"], 3) for k, v in pd.items()})
    print("all tests passed")


if __name__ == "__main__":
    main()
