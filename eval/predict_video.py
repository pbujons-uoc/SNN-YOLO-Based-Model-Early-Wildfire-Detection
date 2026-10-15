#!/usr/bin/env python3
"""Qualitative video test with the NEW models (one model per process, because SpikeYOLO and YOLOv8m need different
ultralytics versions). Every frame is treated independently (single image, same 640x640 letterbox, same network input and the
same per-model confidence threshold as the test-set evaluation). No sliding window: the new input layer handles one image.

    python predict_video.py --model yolov8m_s0            --videos ../videos
    python predict_video.py --model spikeyolo_rgb_T4_s0   --videos ../videos
    python predict_video.py --model spikeyolo_poisson_T4_s0 --videos ../videos
    python predict_video.py --model spikeyolo_latency_T4_s0 --videos ../videos
Optional zero-shot temporal variant (direct and Poisson SpikeYOLO only):
    python predict_video.py --model spikeyolo_poisson_T4_s0 --videos ../videos --window
feeds T CONSECUTIVE frames to the T time steps (frame t-T+1 .. t -> steps 1..T, detections assigned to frame t; the first frames
repeat frame 0). The networks were trained with the same content at all steps, so this is NOT validated, only a qualitative
demonstration. Output goes to <video>/<model>_window/. Latency input is not defined for this variant (its code is a temporal
code inside ONE image).
(--videos is a folder with .mp4/.avi/.mov/.mkv files, or a single video file.)

Output (../Results_test/videos/<video>/<model>/): annotated.mp4 (boxes drawn, smoke blue, fire red) and detections.npz
(per frame: x1 y1 x2 y2 conf cls in ORIGINAL frame pixels). Then run make_video_figures.py (no GPU needed).
With --measure 300 it also writes timing.json (ms/frame and net GPU mJ/frame, batch 1, fp32, idle subtracted, 3 repeats). Run nothing else on the GPU meanwhile.
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

import models as M

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
COLORS_BGR = {0: (180, 119, 31), 1: (40, 39, 214)}      # smoke blue, fire red (BGR)


def letterbox(frame_bgr, size=640):
    """-> RGB float image [3,size,size] in [0,1] (numpy), scale r, pad (dx, dy). Same geometry as Ultralytics LetterBox(auto=False)."""
    h, w = frame_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else frame_bgr
    dx, dy = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dy - 0.1)), int(round(dy + 0.1))
    left, right = int(round(dx - 0.1)), int(round(dx + 0.1))
    im = cv2.copyMakeBorder(im, top, size - nh - top, left, size - nw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return rgb, r, (left, top)


def to_original(boxes, r, pad, shape):
    h, w = shape[:2]
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad[0]) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - pad[1]) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def draw(frame, dets, label, scale):
    out = frame.copy()
    t = max(2, int(round(2 * scale)))
    for x1, y1, x2, y2, conf, cls in dets:
        c = COLORS_BGR[int(cls)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), c, t)
    cv2.rectangle(out, (0, 0), (int(out.shape[1]), int(34 * scale)), (255, 255, 255), -1)
    cv2.putText(out, label, (int(8 * scale), int(25 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (0, 0, 0), max(1, int(2 * scale)), cv2.LINE_AA)
    return out


def process(loaded, thr, video, outdir, device, label, max_width=1280, every=1, window=False):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1.0, max_width / W)
    ow, oh = int(round(W * s)) // 2 * 2, int(round(H * s)) // 2 * 2
    outdir.mkdir(parents=True, exist_ok=True)
    wr = cv2.VideoWriter(str(outdir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / every, (ow, oh))
    allboxes, ptr, i, t0 = [], [0], 0, time.time()
    layer0 = loaded.layer0
    if window:
        T, mode = layer0.T, layer0.mode
        orig_fwd = layer0.forward
        layer0.forward = lambda x: x                      # clip already built: [T,1,3,H,W]
        buf = deque(maxlen=T)
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every:
                i += 1
                continue
            rgb, r, pad = letterbox(frame)
            x = torch.from_numpy(rgb).unsqueeze(0).to(device)
            if window:
                if not buf:
                    buf.extend([x] * T)
                else:
                    buf.append(x)
                fr = torch.stack(list(buf), 0)             # [T,1,3,H,W], step t = frame t
                if mode == "poisson":
                    fr = (torch.rand_like(fr) < fr).to(fr.dtype)
                x = fr
            dets = loaded.decode(loaded.net(x))[0].detach().cpu().numpy().reshape(-1, 6)   # NMS at conf 0.001, as in the evaluation
            d = to_original(dets[dets[:, 4] >= thr], r, pad, frame.shape)
            allboxes.append(d); ptr.append(ptr[-1] + len(d))
            small = cv2.resize(frame, (ow, oh)) if s != 1.0 else frame
            wr.write(draw(small, np.concatenate([d[:, :4] * s, d[:, 4:]], 1), label, ow / 1280))
            i += 1
            if i % 200 == 0:
                print(f"    {i}/{n_total} frames  ({(time.time() - t0):.0f}s)", flush=True)
    cap.release(); wr.release()
    if window:
        layer0.forward = orig_fwd
    allb = np.concatenate(allboxes) if allboxes else np.zeros((0, 6))
    np.savez_compressed(outdir / "detections.npz", ptr=np.array(ptr), dets=allb, fps=fps, width=W, height=H, every=every, threshold=thr)
    n = len(allboxes)
    cls = [set(b[:, 5].astype(int).tolist()) for b in allboxes]
    print(f"  {video.name}: {n} frames, frames with smoke {sum(0 in c for c in cls)}, with fire {sum(1 in c for c in cls)}")
    return n


def measure(loaded, video, outdir, device, n=300, idle_s=10.0, warmup=30, repeats=3):
    """Time and GPU energy of the full forward pass (input layer + network + NMS) at batch 1, fp32, on n frames of this video
    spread evenly over it, preloaded on the GPU and run back to back in one window, minus the idle power measured before.
    Same logic as the benchmark of the paper (NVML total-energy counter), on video frames."""
    try:
        import pynvml
        pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(device.index or 0))
        en = lambda: float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))      # mJ
    except Exception as e:
        print("   NVML not available, timing without energy:", e); en = None
    cap = cv2.VideoCapture(str(video)); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pool = []
    for k in np.linspace(0, max(N - 2, 0), min(n, N)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); ok, f = cap.read()
        if ok:
            pool.append(torch.from_numpy(letterbox(f)[0]).unsqueeze(0).to(device))
    cap.release()
    P = len(pool)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()
    def window(seconds=None):
        sync(); e0, t0 = (en() if en else 0.0), time.perf_counter(); i = 0
        while True:
            with torch.no_grad():
                loaded.decode(loaded.net(pool[i % P]))
            i += 1
            if seconds is None and i >= P * 3:
                break
            if seconds is not None and time.perf_counter() - t0 > seconds:
                break
        sync(); t1 = time.perf_counter(); e1 = en() if en else 0.0
        return i, t1 - t0, e1 - e0
    with torch.no_grad():
        for i in range(warmup):
            loaded.decode(loaded.net(pool[i % P]))
    sync()
    res = []
    for r in range(repeats):
        if en:
            time.sleep(2.0)
            t0, e0 = time.perf_counter(), en(); time.sleep(idle_s); idle_W = (en() - e0) / 1000.0 / (time.perf_counter() - t0)
        else:
            idle_W = 0.0
        it, dt, de = window()
        res.append(dict(frames=it, ms_per_frame=1000 * dt / it, fps=it / dt, gpu_power_W=(de / 1000.0 / dt) if en else None,
                        idle_W=idle_W, mJ_per_frame_gross=(de / it) if en else None,
                        mJ_per_frame_net=((de - idle_W * 1000.0 * dt) / it) if en else None))
    out = dict(video=video.name, pool_frames=P, repeats=res,
               ms_per_frame_mean=float(np.mean([x["ms_per_frame"] for x in res])),
               mJ_per_frame_net_mean=float(np.mean([x["mJ_per_frame_net"] for x in res])) if en else None)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "timing.json").write_text(json.dumps(out, indent=1))
    print(f"  timing {video.name}: {out['ms_per_frame_mean']:.2f} ms/frame, net GPU energy "
          f"{out['mJ_per_frame_net_mean'] if en else float('nan'):.1f} mJ/frame (idle {res[0]['idle_W']:.1f} W)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="run folder name, e.g. spikeyolo_poisson_T4_s0")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default=str(M.RESULTS_TEST / "videos"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--every", type=int, default=1, help="process every N-th frame (1 = all)")
    ap.add_argument("--measure", type=int, default=0, metavar="N", help="also time the forward pass and measure GPU energy on N frames of each video (per-frame mode only; N=300 is enough)")
    ap.add_argument("--window", action="store_true", help="zero-shot: T consecutive frames -> T time steps (direct/Poisson SpikeYOLO)")
    ap.add_argument("--seed", type=int, default=0, help="generator seed (matters only for the Poisson input)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = M.device_from(a.device)
    spec = M.get_spec(a.model)
    thr = json.loads((M.RESULTS_TEST / a.model / "metrics_seed0.json").read_text())["threshold"]
    loaded = M.load(spec, device)
    nice = {"rgb": "SpikeYOLO direct", "poisson": "SpikeYOLO Poisson", "latency": "SpikeYOLO latency"}
    label = (nice[spec.encoding] + f" (T={spec.T})") if spec.family == "spikeyolo" else {"yolov8": "YOLOv8m", "vanilla": "VanillaCNN"}[spec.family]
    if a.window:
        if spec.family != "spikeyolo" or spec.encoding == "latency":
            raise SystemExit("--window is only defined for direct and Poisson SpikeYOLO")
        label += " (window)"
    print(f"{a.model}: threshold {thr}, label '{label}'")
    src = Path(a.videos)
    vids = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise SystemExit(f"no videos found in {src}")
    for v in vids:
        print(" ", v.name)
        if a.measure and not a.window:
            measure(loaded, v, Path(a.out) / v.stem / a.model, device, a.measure)
        process(loaded, thr, v, Path(a.out) / v.stem / (a.model + ("_window" if a.window else "")), device, label, a.max_width, a.every, a.window)


if __name__ == "__main__":
    main()