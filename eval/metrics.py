#!/usr/bin/env python3
"""Detection metrics for the unified test-set evaluation. Pure numpy (no torch / ultralytics), so the
numbers do not depend on which ultralytics version produced the predictions.

Input : a prediction dump written by dump_predictions.py (per image: detections after NMS at conf >= 0.001,
        and the ground-truth boxes, both in pixels of the 640 x 640 letterboxed image).
Output: metrics.json with
  * mAP50, mAP50-95 and per-class AP (Ultralytics definition: 101-point interpolated AP, matching identical to
    the Ultralytics validator),
  * a confidence threshold chosen on the VALIDATION dump (max of the class-averaged F1 at IoU 0.5) and
    TP / FP / FN / precision / recall / F1 on the TEST dump at that threshold (matching redone on the
    thresholded detections),
  * recall by object size, false-alarm rate on images without annotations,
  * bootstrap 95 % intervals over test images.

    python metrics.py --val ../Results_test/<model>/val_seed0.npz --test ../Results_test/<model>/test_seed0.npz \
                      --out ../Results_test/<model>/metrics_seed0.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

IOUV = np.linspace(0.5, 0.95, 10)
SIZE_BINS = [(0, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, np.inf, ">=64")]   # sqrt(area) in 640-px units
DEFAULT_NAMES = ["smoke", "fire"]                                                           # D-Fire_clean: 0 smoke, 1 fire
THR_GRID = np.round(np.arange(0.01, 0.951, 0.01), 2)


# --------------------------------------------------------------------------------------------- dump I/O
class Dump:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.path = str(path)
        self.files = d["files"]
        self.det_ptr, self.dets = d["det_ptr"], d["dets"]      # dets: (N, 6) x1 y1 x2 y2 conf cls
        self.gt_ptr, self.gts = d["gt_ptr"], d["gts"]          # gts : (M, 5) cls x1 y1 x2 y2
        self.n = len(self.files)

    def det(self, i):
        return self.dets[self.det_ptr[i]:self.det_ptr[i + 1]]

    def gt(self, i):
        return self.gts[self.gt_ptr[i]:self.gt_ptr[i + 1]]


# --------------------------------------------------------------------------------------------- matching
def box_iou(a, b):
    """a (m,4), b (n,4) xyxy -> (m,n)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match_core(iou, same, iouv):
    """Ultralytics' matching (DetectionValidator.match_predictions), numpy port.
    iou, same: (m_gt, n_det). Returns correct (n_det, len(iouv)) and gt_hit (m_gt, len(iouv))."""
    m, n = iou.shape
    correct = np.zeros((n, len(iouv)), dtype=bool)
    hit = np.zeros((m, len(iouv)), dtype=bool)
    if m == 0 or n == 0:
        return correct, hit
    for k, thr in enumerate(iouv):
        x = np.nonzero((iou >= thr) & same)
        if x[0].shape[0]:
            mt = np.concatenate([np.stack(x, 1), iou[x[0], x[1]][:, None]], 1)
            if x[0].shape[0] > 1:
                mt = mt[mt[:, 2].argsort()[::-1]]
                mt = mt[np.unique(mt[:, 1], return_index=True)[1]]
                mt = mt[np.unique(mt[:, 0], return_index=True)[1]]
            correct[mt[:, 1].astype(int), k] = True
            hit[mt[:, 0].astype(int), k] = True
    return correct, hit


def match(det, gt, iouv=IOUV):
    iou = box_iou(gt[:, 1:5], det[:, :4])
    same = gt[:, 0:1] == det[None, :, 5] if len(gt) and len(det) else np.zeros((len(gt), len(det)), bool)
    return match_core(iou, same, iouv)


# --------------------------------------------------------------------------------------------- AP
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(getattr(np, "trapezoid", getattr(np, "trapz", None))(y, x))


def ap_weighted(correct, cls, wdet, n_gt):
    """AP per class and IoU. correct (N,10) bool and cls (N,), already sorted by decreasing confidence; wdet (N,)
    weight of each detection (1 = plain; bootstrap multiplicity otherwise); n_gt (nc,) weighted GT counts.
    With unit weights this is exactly the Ultralytics ap_per_class (without the P/R curves)."""
    nc = len(n_gt)
    ap = np.zeros((nc, correct.shape[1]))
    for c in range(nc):
        m = cls == c
        if n_gt[c] <= 0 or not m.any():
            continue
        wc = wdet[m][:, None]
        tpc = np.cumsum(correct[m] * wc, 0)
        fpc = np.cumsum((~correct[m]) * wc, 0)
        recall = tpc / (n_gt[c] + 1e-16)
        precision = tpc / np.maximum(tpc + fpc, 1e-16)
        for k in range(correct.shape[1]):
            ap[c, k] = compute_ap(recall[:, k], precision[:, k])
    return ap


def prf(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1e-16)
    r = tp / np.maximum(tp + fn, 1e-16)
    f = 2 * p * r / np.maximum(p + r, 1e-16)
    return p, r, f


# --------------------------------------------------------------------------------------------- threshold
def select_threshold(dump, nc=2, grid=THR_GRID):
    """Confidence threshold that maximises the class-averaged F1 at IoU 0.5 on `dump` (the validation set)."""
    pre = []
    for i in range(dump.n):
        d, g = dump.det(i), dump.gt(i)
        iou = box_iou(g[:, 1:5], d[:, :4])
        same = (g[:, 0:1] == d[None, :, 5]) if len(g) and len(d) else np.zeros((len(g), len(d)), bool)
        pre.append((iou, same, d[:, 4], d[:, 5].astype(int), g[:, 0].astype(int)))
    curve = []
    for thr in grid:
        tp = np.zeros(nc); fp = np.zeros(nc); fn = np.zeros(nc)
        for iou, same, conf, dcls, gcls in pre:
            keep = conf >= thr
            correct, _ = match_core(iou[:, keep], same[:, keep], IOUV[:1])
            dc = dcls[keep]
            for c in range(nc):
                t = int((correct[:, 0] & (dc == c)).sum())
                tp[c] += t
                fp[c] += int((dc == c).sum()) - t
                fn[c] += int((gcls == c).sum()) - t
        _, _, f = prf(tp, fp, fn)
        curve.append(float(f.mean()))
    best = int(np.argmax(curve))
    return float(grid[best]), {"thr": [float(t) for t in grid], "f1_mean_class": curve}


# --------------------------------------------------------------------------------------------- evaluator
class Eval:
    """Everything needed to compute the metrics of one model on one split, for any set of image weights."""

    def __init__(self, dump, thr, names=DEFAULT_NAMES):
        self.dump, self.thr, self.names, self.nc, self.n = dump, thr, list(names), len(names), dump.n
        det_img, corr, conf, cls = [], [], [], []
        tp_img = np.zeros((self.n, self.nc)); fp_img = np.zeros((self.n, self.nc))
        gtc = np.zeros((self.n, self.nc)); self.det_over = np.zeros(self.n, bool)
        g_img, g_cls, g_size, g_hit = [], [], [], []
        for i in range(self.n):
            d, g = dump.det(i), dump.gt(i)
            c_all, _ = match(d, g, IOUV)
            det_img.append(np.full(len(d), i)); corr.append(c_all); conf.append(d[:, 4]); cls.append(d[:, 5].astype(int))
            gcls = g[:, 0].astype(int)
            for c in range(self.nc):
                gtc[i, c] = (gcls == c).sum()
            dk = d[d[:, 4] >= thr]
            c05, hit = match(dk, g, IOUV[:1])
            dkc = dk[:, 5].astype(int)
            for c in range(self.nc):
                t = (c05[:, 0] & (dkc == c)).sum()
                tp_img[i, c] = t
                fp_img[i, c] = (dkc == c).sum() - t
            self.det_over[i] = len(dk) > 0
            size = np.sqrt(np.clip((g[:, 3] - g[:, 1]) * (g[:, 4] - g[:, 2]), 0, None)) if len(g) else np.zeros(0)
            g_img.append(np.full(len(g), i)); g_cls.append(gcls); g_size.append(size); g_hit.append(hit[:, 0] if len(g) else np.zeros(0, bool))
        cat = lambda xs, dt: np.concatenate(xs) if xs and sum(len(x) for x in xs) else np.zeros((0,) + xs[0].shape[1:], dt)
        det_img, conf, cls = cat(det_img, int), cat(conf, float), cat(cls, int)
        corr = np.concatenate(corr, 0) if len(corr) else np.zeros((0, 10), bool)
        o = np.argsort(-conf, kind="stable")                       # sort once, by decreasing confidence
        self.det_img, self.correct, self.cls = det_img[o], corr[o], cls[o]
        self.tp_img, self.fp_img, self.gt_cnt = tp_img, fp_img, gtc
        self.g_img, self.g_cls, self.g_size, self.g_hit = cat(g_img, int), cat(g_cls, int), cat(g_size, float), cat(g_hit, bool)
        self.empty = gtc.sum(1) == 0

    def metrics(self, w=None):
        w = np.ones(self.n) if w is None else np.asarray(w, float)
        n_gt = (w[:, None] * self.gt_cnt).sum(0)
        ap = ap_weighted(self.correct, self.cls, w[self.det_img], n_gt)
        out = {"map50": float(ap[:, 0].mean()), "map50_95": float(ap.mean())}
        for c, nm in enumerate(self.names):
            out[f"ap50_{nm}"] = float(ap[c, 0]); out[f"ap50_95_{nm}"] = float(ap[c].mean())
        tp = (w[:, None] * self.tp_img).sum(0); fp = (w[:, None] * self.fp_img).sum(0); fn = n_gt - tp
        p, r, f = prf(tp, fp, fn)
        P, R, F = prf(tp.sum(), fp.sum(), fn.sum())
        out.update(precision=float(P), recall=float(R), f1=float(F), f1_macro=float(f.mean()),
                   tp=float(tp.sum()), fp=float(fp.sum()), fn=float(fn.sum()))
        for c, nm in enumerate(self.names):
            out.update({f"precision_{nm}": float(p[c]), f"recall_{nm}": float(r[c]), f"f1_{nm}": float(f[c]),
                        f"tp_{nm}": float(tp[c]), f"fp_{nm}": float(fp[c]), f"fn_{nm}": float(fn[c])})
        we = w * self.empty
        out["false_alarm_rate_empty_images"] = float((we * self.det_over).sum() / we.sum()) if we.sum() > 0 else float("nan")
        wg = w[self.g_img]
        for lo, hi, lab in SIZE_BINS:
            for c, nm in enumerate(self.names):
                m = (self.g_cls == c) & (self.g_size >= lo) & (self.g_size < hi)
                tot = (wg * m).sum()
                out[f"recall_{nm}_size{lab}"] = float((wg * m * self.g_hit).sum() / tot) if tot > 0 else float("nan")
                out[f"n_gt_{nm}_size{lab}"] = float(tot)
        return out


def bootstrap(evals, n_boot=1000, seed=0):
    """Image-level bootstrap shared by all evaluators (they must contain the same images in the same order).
    Returns, for each evaluator, {metric: array(n_boot)} computed with multiplicity weights (equivalent to
    resampling the images with replacement)."""
    n = evals[0].n
    assert all(e.n == n for e in evals), "evaluators have a different number of images"
    rng = np.random.default_rng(seed)
    res = [dict() for _ in evals]
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k, e in enumerate(evals):
            for key, v in e.metrics(w).items():
                res[k].setdefault(key, []).append(v)
    return [{k: np.asarray(v) for k, v in r.items()} for r in res]


def ci(a, level=0.95):
    a = a[~np.isnan(a)]
    if not len(a):
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return [float(lo), float(hi)]


def paired_difference(ev_a, ev_b, keys=("map50", "map50_95", "f1"), n_boot=1000, seed=0):
    """A - B on the same bootstrap resamples. Needs the same image order in both dumps."""
    assert np.array_equal(ev_a.dump.files, ev_b.dump.files), "dumps do not contain the same images in the same order"
    pa, pb = ev_a.metrics(), ev_b.metrics()
    ba, bb = bootstrap([ev_a, ev_b], n_boot, seed)
    out = {}
    for k in keys:
        d = ba[k] - bb[k]
        out[k] = {"diff": pa[k] - pb[k], "ci95": ci(d), "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
    return out


# --------------------------------------------------------------------------------------------- CLI
def evaluate(val_path, test_path, out_path=None, threshold=None, n_boot=1000, names=DEFAULT_NAMES, seed=0):
    val = Dump(val_path)
    test = Dump(test_path)
    curve = None
    if threshold is None:
        threshold, curve = select_threshold(val, len(names))
    ev = Eval(test, threshold, names)
    point = ev.metrics()
    res = {"val_dump": str(val_path), "test_dump": str(test_path), "threshold": threshold, "n_test_images": test.n,
           "names": list(names), "point": point, "f1_curve_val": curve}
    if n_boot:
        b = bootstrap([ev], n_boot, seed)[0]
        res["bootstrap95"] = {k: ci(v) for k, v in b.items()}
        res["bootstrap_n"] = n_boot
    if out_path:
        Path(out_path).write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True, help="validation dump (chooses the confidence threshold)")
    ap.add_argument("--test", required=True, help="test dump")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="fix the threshold instead of choosing it on val")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES, help="class names in index order (D-Fire_clean: smoke fire)")
    a = ap.parse_args()
    r = evaluate(a.val, a.test, a.out, a.threshold, a.n_boot, a.names)
    p = r["point"]
    print(f"threshold {r['threshold']:.2f} | mAP50 {p['map50']:.4f} mAP50-95 {p['map50_95']:.4f} | "
          f"P {p['precision']:.3f} R {p['recall']:.3f} F1 {p['f1']:.3f} | TP {p['tp']:.0f} FP {p['fp']:.0f} FN {p['fn']:.0f}")


if __name__ == "__main__":
    main()
