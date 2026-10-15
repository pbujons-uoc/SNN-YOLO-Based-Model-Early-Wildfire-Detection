#!/usr/bin/env python3
"""Looks for near-duplicate images ACROSS the train / val / test splits (data leakage). CPU only, no GPU needed.

Method (two stages, so that flat sky/smoke images do not give false alarms):
  1. dHash: every image -> grayscale 9x8 -> 64-bit hash of horizontal gradients. Hamming distance <= --max-hamming
     (default 8) makes a candidate pair.
  2. Verification: normalised cross-correlation (NCC) of the 32x32 grayscale thumbnails of the candidate pair.
     A pair is a CONFIRMED near-duplicate if hamming <= --max-hamming and NCC >= --min-ncc (default 0.95).
     Exact copies, re-compressed copies, slightly resized or cropped copies and consecutive video frames with little
     motion are all caught. It will not catch a very different view of the same scene.

    python check_leakage.py                                   # all three splits
    python check_leakage.py --data-root ../datasets/D-Fire_clean --out ../Results_test/leakage

Output (in --out):
  leakage_summary.json / printed table   how many val / test images have a confirmed duplicate in the reference split(s)
  leakage_<query>_vs_<reference>.csv     one row per query image with a candidate: nearest image, hamming, NCC, confirmed
  leakage_test_confirmed.txt             test images that have a confirmed duplicate in train or val (to re-evaluate without them)
  leakage_examples.png                   side by side examples (query | nearest reference) to look at with your own eyes
  hashes.npz                             cache (delete it if the images change)
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def describe(path):
    """-> (64-bit dHash as uint64, 32x32 thumbnail as float32 vector, zero mean / unit norm)"""
    try:
        im = Image.open(path)
        im.draft("L", (256, 256))                        # fast JPEG decode at reduced size
        im = im.convert("L")
        g = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.float32)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        h = np.packbits(bits).view(np.uint64)[0]
        t = np.asarray(im.resize((32, 32), Image.LANCZOS), dtype=np.float32).flatten()
        t = t - t.mean()
        t = t / (np.linalg.norm(t) + 1e-6)
        return h, t.astype(np.float32)
    except Exception as e:                               # unreadable image: marked with an impossible hash
        print(f"  could not read {path}: {e}", flush=True)
        return np.uint64(0), np.zeros(1024, np.float32)


def list_images(split_dir):
    d = split_dir / "images"
    d = d if d.exists() else split_dir
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in EXT)


def hamming_matrix(q, r):
    x = np.bitwise_xor(q[:, None], r[None, :])                           # (nq, nr) uint64
    return POP[x.view(np.uint8).reshape(len(q), len(r), 8)].sum(-1)      # (nq, nr)


def compare(q, r, max_ham, min_ncc, chunk=256):
    """for every query image: nearest reference by hamming, and the best NCC among candidates with hamming <= max_ham"""
    qh, qt = q["hash"], q["thumb"]
    rh, rt = r["hash"], r["thumb"]
    rows = []
    for a in range(0, len(qh), chunk):
        D = hamming_matrix(qh[a:a + chunk], rh)
        for k in range(D.shape[0]):
            i = a + k
            d = D[k]
            j_near = int(np.argmin(d))
            cand = np.where(d <= max_ham)[0]
            if len(cand) > 200:                                           # very flat image: keep the 200 closest
                cand = cand[np.argsort(d[cand])[:200]]
            best_ncc, best_j = -1.0, j_near
            if len(cand):
                ncc = rt[cand] @ qt[i]
                m = int(np.argmax(ncc))
                best_ncc, best_j = float(ncc[m]), int(cand[m])
            rows.append(dict(query=q["paths"][i], nearest=r["paths"][best_j], hamming=int(d[best_j]),
                             min_hamming=int(d[j_near]), n_candidates=int(len(cand)), ncc=best_ncc,
                             confirmed=bool(len(cand) and d[best_j] <= max_ham and best_ncc >= min_ncc)))
    return rows


def montage(pairs, out, n=24, size=160):
    pairs = pairs[:n]
    if not pairs:
        return
    cols = 3
    rows_n = (len(pairs) + cols - 1) // cols
    W, H = 2 * size + 20, size + 28
    sheet = Image.new("RGB", (cols * W, rows_n * H), "white")
    dr = ImageDraw.Draw(sheet)
    for n_, p in enumerate(pairs):
        x0, y0 = (n_ % cols) * W, (n_ // cols) * H
        for k, key in enumerate(("query", "nearest")):
            try:
                im = Image.open(p[key]).convert("RGB")
                im.thumbnail((size, size))
                sheet.paste(im, (x0 + k * (size + 10), y0 + 14))
            except Exception:
                pass
        dr.text((x0 + 2, y0 + 1), f"{Path(p['query']).name} | {Path(p['nearest']).parent.parent.name}/{Path(p['nearest']).name}"
                                  f"  d={p['hamming']} ncc={p['ncc']:.3f}", fill="black")
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(HERE.parent.parent / "datasets" / "D-Fire_clean"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-hamming", type=int, default=8)
    ap.add_argument("--min-ncc", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HERE.parent / "Results_test" / "leakage"))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "hashes.npz"
    S = {}
    cached = np.load(cache, allow_pickle=False) if (cache.exists() and not a.rebuild) else None
    for s in a.splits:
        if cached is not None and f"{s}_paths" in cached.files:
            S[s] = dict(paths=[str(p) for p in cached[f"{s}_paths"]], hash=cached[f"{s}_hash"], thumb=cached[f"{s}_thumb"])
            print(f"{s}: {len(S[s]['paths'])} images (cached)")
            continue
        paths = list_images(root / s)
        print(f"{s}: hashing {len(paths)} images ...", flush=True)
        with Pool(a.workers) as pool:
            res = pool.map(describe, [str(p) for p in paths], chunksize=64)
        S[s] = dict(paths=[str(p) for p in paths], hash=np.array([r[0] for r in res], dtype=np.uint64),
                    thumb=np.stack([r[1] for r in res]))
    np.savez_compressed(cache, **{f"{s}_{k}": (np.array(v["paths"]) if k == "paths" else v[k])
                                  for s, v in S.items() for k in ("paths", "hash", "thumb")})

    plan = [("val", ["train"]), ("test", ["train"]), ("test", ["val"]), ("test", ["train", "val"])]
    summary, confirmed_test, all_pairs = {}, set(), []
    for qs, rs in plan:
        if qs not in S or any(r not in S for r in rs):
            continue
        ref = dict(paths=sum((S[r]["paths"] for r in rs), []), hash=np.concatenate([S[r]["hash"] for r in rs]),
                   thumb=np.concatenate([S[r]["thumb"] for r in rs]))
        rows = compare(S[qs], ref, a.max_hamming, a.min_ncc)
        tag = f"{qs}_vs_{'+'.join(rs)}"
        with open(out / f"leakage_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n = len(rows)
        conf = [r for r in rows if r["confirmed"]]
        strict = [r for r in conf if r["hamming"] <= 2 and r["ncc"] >= 0.98]
        summary[tag] = dict(n_query=n, n_reference=len(ref["paths"]), n_with_candidate=sum(r["n_candidates"] > 0 for r in rows),
                            n_confirmed_near_duplicate=len(conf), pct_confirmed=100.0 * len(conf) / n,
                            n_strict_duplicate=len(strict), pct_strict=100.0 * len(strict) / n)
        if tag == "test_vs_train+val":
            confirmed_test = {r["query"] for r in conf}
        if tag in ("test_vs_train", "val_vs_train"):
            all_pairs += conf
    (out / "leakage_summary.json").write_text(json.dumps(dict(settings=dict(max_hamming=a.max_hamming, min_ncc=a.min_ncc),
                                                              result=summary), indent=2))
    (out / "leakage_test_confirmed.txt").write_text("\n".join(sorted(confirmed_test)) + ("\n" if confirmed_test else ""))
    all_pairs.sort(key=lambda p: (p["hamming"], -p["ncc"]))
    montage(all_pairs[:12] + all_pairs[len(all_pairs) // 2:len(all_pairs) // 2 + 6] + all_pairs[-6:], out / "leakage_examples.png")

    print(f"\nnear-duplicate check (dHash hamming <= {a.max_hamming} and thumbnail NCC >= {a.min_ncc})")
    print(f"{'query vs reference':22s} {'queries':>8s} {'confirmed':>10s} {'%':>6s} {'strict(d<=2,ncc>=.98)':>22s}")
    for tag, s in summary.items():
        print(f"{tag:22s} {s['n_query']:8d} {s['n_confirmed_near_duplicate']:10d} {s['pct_confirmed']:6.1f} "
              f"{s['n_strict_duplicate']:12d} ({s['pct_strict']:.1f}%)")
    print(f"\nfiles in {out}")
    print("Look at leakage_examples.png before deciding: a small percentage with identical scenes is normal for video-based "
          "datasets; a large one means the splits are not independent.")


if __name__ == "__main__":
    main()