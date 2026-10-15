#!/usr/bin/env python3
"""Time and energy of one trained model. One invocation = one repeat; run_eval.py calls it several times,
interleaving the models, so that GPU temperature and clock drift do not favour any model.

What is measured (GPU only; the CPU and RAM are not included):
  network    forward pass of the detector, input already encoded and resident in GPU memory, NO NMS.
             For SpikeYOLO the layer-0 encoding is bypassed: this is the number that excludes the encoding.
  encode     layer 0 alone (repeat / Bernoulli / latency), SpikeYOLO only.
  endtoend   uint8 image on the GPU -> /255 -> (encoding) -> network -> NMS.
  idle       GPU idle power with the model loaded, before and after (subtracted to give the net energy).
Batch size 1 by default (one frame at a time, as on a UAV); every iteration is synchronised, so the latency
distribution is that of a closed loop. Energy comes from the NVML cumulative energy counter of the GPU
(mJ), with a 10 Hz power trace as a cross-check; power, temperature and clocks are logged.
CodeCarbon (machine mode: GPU + CPU + RAM, offline with a fixed country so the grid factor is reproducible) runs around
the very same windows and gives the CO2 figure; its keys in the output start with cc_. The NVML number is GPU only and is
the one used to compare models; CodeCarbon is kept for continuity with the first version of the paper and for CO2.

    python benchmark_efficiency.py --model spikeyolo_rgb_T4_s0 --repeat 1
Never run it while a training is using the GPU (the script refuses unless --allow-busy).
"""
import argparse
import copy
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import torch

import models as M

try:
    import pynvml
    pynvml.nvmlInit()
    HAVE_NVML = True
except Exception:                                     # CPU-only test machine
    pynvml, HAVE_NVML = None, False


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def energy_mj(h):
    try:
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
    except Exception:
        return None


class Sampler(threading.Thread):
    """10 Hz power / temperature / clock trace."""

    def __init__(self, h, hz=10.0):
        super().__init__(daemon=True)
        self.h, self.dt, self.stop_flag, self.rows = h, 1.0 / hz, threading.Event(), []

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.rows.append((time.perf_counter(),
                                  pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                                  pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                                  pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                                  pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu))
            except Exception:
                pass
            self.stop_flag.wait(self.dt)

    def finish(self):
        self.stop_flag.set()
        self.join()
        r = np.array(self.rows, dtype=float).reshape(-1, 5)
        return r


CC = {"country": "ESP", "gpu_id": 0, "enabled": True, "error": None}


def cc_start():
    if not CC["enabled"]:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker
        base = dict(country_iso_code=CC["country"], measure_power_secs=1, save_to_file=False,
                    gpu_ids=[CC["gpu_id"]], project_name="eval_benchmark")
        # the name of the log argument changed between CodeCarbon versions (log_level in 3.x, logging_level before)
        t = None
        for extra in ({"log_level": "error"}, {"logging_level": "error"}, {}):
            try:
                t = OfflineEmissionsTracker(**base, **extra)
                break
            except TypeError as e:
                CC["error"] = f"TypeError: {e}"
        if t is None:
            return None
        CC["error"] = None
        t.start()
        return t
    except Exception as e:                                  # CodeCarbon missing / failed: keep going without it
        CC["error"] = f"{type(e).__name__}: {e}"
        return None


def cc_stop(t):
    if t is None:
        return None
    try:
        t.stop()
        d = t.final_emissions_data
        return dict(kWh=float(d.energy_consumed), cpu_kWh=float(d.cpu_energy), gpu_kWh=float(d.gpu_energy),
                    ram_kWh=float(d.ram_energy), co2_kg=float(d.emissions), cpu_W=float(d.cpu_power),
                    gpu_W=float(d.gpu_power), ram_W=float(d.ram_power))
    except Exception as e:
        CC["error"] = f"{type(e).__name__}: {e}"
        return None


def summarise(label, lat_ms, seconds, batch, e0, e1, trace, idle_w, cc=None, cc_idle_w=None):
    n = len(lat_ms)
    out = {"window": label, "n_iters": n, "batch": batch, "seconds": seconds,
           "latency_ms_mean": float(np.mean(lat_ms)) if n else None,
           "latency_ms_median": float(np.median(lat_ms)) if n else None,
           "latency_ms_std": float(np.std(lat_ms, ddof=1)) if n > 1 else None,
           "latency_ms_p95": float(np.percentile(lat_ms, 95)) if n else None,
           "latency_ms_p99": float(np.percentile(lat_ms, 99)) if n else None,
           "throughput_img_s": n * batch / seconds if n else None}
    e_counter = (e1 - e0) / 1000.0 if (e0 is not None and e1 is not None) else None
    e_integr = None
    if len(trace) > 1:
        tt, pw = trace[:, 0], trace[:, 1]
        e_integr = float(np.sum((pw[1:] + pw[:-1]) / 2 * np.diff(tt)))
        out.update(power_mean_W_trace=float(pw.mean()), temp_C_start=float(trace[0, 2]), temp_C_end=float(trace[-1, 2]),
                   sm_clock_MHz_mean=float(trace[:, 3].mean()), gpu_util_pct_mean=float(trace[:, 4].mean()))
    e_j = e_counter if e_counter is not None else e_integr
    out.update(energy_counter_J=e_counter, energy_integrated_J=e_integr, energy_basis="counter" if e_counter is not None else "trace")
    if e_j is not None and n:
        out["power_mean_W"] = e_j / seconds
        out["energy_gross_mJ_per_image"] = e_j * 1000.0 / (n * batch)
        if idle_w is not None:
            out["energy_net_mJ_per_image"] = (e_j - idle_w * seconds) * 1000.0 / (n * batch)
    if cc is not None and n:
        j = cc["kWh"] * 3.6e6                                  # kWh -> J
        out.update(cc_energy_kWh=cc["kWh"], cc_cpu_kWh=cc["cpu_kWh"], cc_gpu_kWh=cc["gpu_kWh"], cc_ram_kWh=cc["ram_kWh"],
                   cc_co2_kg=cc["co2_kg"], cc_power_mean_W=j / seconds,
                   cc_energy_gross_mJ_per_image=j * 1000.0 / (n * batch),
                   cc_co2_mg_per_image_gross=cc["co2_kg"] * 1e6 / (n * batch))
        if cc_idle_w is not None and j > 0:
            net_j = j - cc_idle_w * seconds
            out["cc_energy_net_mJ_per_image"] = net_j * 1000.0 / (n * batch)
            out["cc_co2_mg_per_image_net"] = cc["co2_kg"] * (net_j / j) * 1e6 / (n * batch)   # same grid factor
    return out


def run_window(label, fn, h, iters, min_seconds, batch, idle_w=None, idle_seconds=None, cc_idle_w=None):
    """fn(i) runs one iteration (the caller indexes its own pool). idle window if fn is None."""
    sampler = Sampler(h) if HAVE_NVML else None
    if sampler:
        sampler.start()
    cc_tracker = cc_start()
    sync()
    e0, t0, lat = (energy_mj(h) if HAVE_NVML else None), time.perf_counter(), []
    if fn is None:
        time.sleep(idle_seconds)
    else:
        i = 0
        while i < iters or (time.perf_counter() - t0) < min_seconds:
            s = time.perf_counter()
            fn(i)
            sync()
            lat.append((time.perf_counter() - s) * 1000.0)
            i += 1
    sync()
    t1, e1 = time.perf_counter(), (energy_mj(h) if HAVE_NVML else None)
    cc = cc_stop(cc_tracker)
    trace = sampler.finish() if sampler else np.zeros((0, 5))
    if fn is None:                                    # idle: only power matters
        e_j = (e1 - e0) / 1000.0 if (e0 is not None and e1 is not None) else None
        if e_j is None and len(trace) > 1:
            e_j = float(np.sum((trace[1:, 1] + trace[:-1, 1]) / 2 * np.diff(trace[:, 0])))
        return {"window": label, "seconds": t1 - t0, "power_mean_W": e_j / (t1 - t0) if e_j is not None else None,
                "cc_power_mean_W": (cc["kWh"] * 3.6e6 / (t1 - t0)) if cc else None,
                "temp_C_end": float(trace[-1, 2]) if len(trace) else None}, np.array([])
    return summarise(label, lat, t1 - t0, batch, e0, e1, trace, idle_w, cc, cc_idle_w), np.array(lat)


def try_fuse(net, x_ref, mode):
    """Conv+BN fusion (what Ultralytics does for inference) if it exists and leaves the output unchanged."""
    if mode == "no" or not hasattr(net, "fuse"):
        return net, "not fused"
    nbn = lambda m: sum(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in m.modules())
    try:
        cand = copy.deepcopy(net).fuse(verbose=False)
        with torch.no_grad():
            a, b = net(x_ref), cand(x_ref)
        a = a[0] if isinstance(a, (list, tuple)) else a
        b = b[0] if isinstance(b, (list, tuple)) else b
        diff = float((a.float() - b.float()).abs().max())
        info = f"BatchNorm layers {nbn(net)} -> {nbn(cand)}, max |diff| {diff:.1e}"
        if diff < 5e-3:
            return cand.eval(), f"fused ({info})"
        return net, f"fusion rejected ({info})"
    except Exception as e:
        return net, f"fusion failed: {type(e).__name__}: {str(e)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="index of this repeat (only used in the file name)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    ap.add_argument("--pool", type=int, default=32, help="number of distinct test images (batches) cycled through")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--min-seconds", type=float, default=30.0, help="each timed window lasts at least this long")
    ap.add_argument("--idle-seconds", type=float, default=20.0)
    ap.add_argument("--fuse", default="auto", choices=["auto", "no"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-codecarbon", action="store_true", help="skip the CodeCarbon measurement")
    ap.add_argument("--cc-country", default="ESP", help="ISO code of the country whose grid factor CodeCarbon uses (offline mode)")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--out", default=str(M.RESULTS_TEST))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    spec = M.get_spec(a.model)
    out_dir = Path(a.out) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"bench_{a.precision}_b{a.batch}_r{a.repeat}"
    if (out_dir / f"{stem}.json").exists() and not a.force:
        print(f"[skip] {out_dir / stem}.json exists")
        return

    device = M.device_from(a.device)
    h = pynvml.nvmlDeviceGetHandleByIndex(int(a.device)) if (HAVE_NVML and device.type == "cuda") else None
    if h is not None and not a.allow_busy:
        busy = [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)]
        if busy:
            raise SystemExit(f"GPU is in use by process(es) {busy}; energy and time would be meaningless. "
                             "Wait for the training queue to finish (or --allow-busy for a dry run).")

    CC.update(country=a.cc_country, gpu_id=int(a.device) if str(a.device).isdigit() else 0, enabled=not a.no_codecarbon)
    torch.backends.cudnn.benchmark = True
    L = M.load(spec, device)
    loader, _ = M.build_loader(L.U, a.split, a.batch, workers=2, cache=False)
    pool_u8 = []
    for batch in loader:
        pool_u8.append(batch["img"].to(device))
        if len(pool_u8) >= a.pool:
            break
    P = len(pool_u8)
    f32 = [x.float() / 255 for x in pool_u8]
    net, fuse_note = try_fuse(L.net, f32[0], a.fuse)
    layer0 = next((m for m in net.modules() if m.__class__.__name__ == "MS_GetT"), None) if spec.family == "spikeyolo" else None
    orig_forward = layer0.forward if layer0 is not None else None
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=(a.precision == "amp" and device.type == "cuda"))

    def forward(x):
        with torch.no_grad(), amp:
            return net(x)

    print(f"[{spec.name}] params {L.params / 1e6:.2f}M  {fuse_note}  pool {P} x batch {a.batch}  precision {a.precision}")
    windows, lat_all = {}, {}
    bypass_check = None

    def _first(o):                                          # network output -> the main prediction tensor
        while isinstance(o, (list, tuple)):
            o = o[0]
        return o

    idle_before, _ = run_window("idle_before", None, h, 0, 0, a.batch, idle_seconds=a.idle_seconds)
    idle_w = idle_before["power_mean_W"]
    cc_idle_w = idle_before.get("cc_power_mean_W")
    if CC["enabled"] and cc_idle_w is None:
        print(f"  !! CodeCarbon gave no measurement ({CC['error']}); cc_* values will be missing. "
              "Use --no-codecarbon to silence this.", flush=True)

    def go(label, fn):
        for i in range(a.warmup):
            fn(i)
        sync()
        w, lat = run_window(label, fn, h, a.iters, a.min_seconds, a.batch, idle_w, cc_idle_w=cc_idle_w)
        windows[label], lat_all[label] = w, lat
        print(f"  {label:9s} {w['latency_ms_mean']:.2f} ms/iter (p95 {w['latency_ms_p95']:.2f})  "
              f"{w.get('energy_net_mJ_per_image', float('nan')):.1f} mJ/img net (GPU)  {w.get('power_mean_W', float('nan')):.0f} W  "
              f"CC {w.get('cc_co2_mg_per_image_gross', float('nan')):.4f} mg CO2/img", flush=True)

    # network only (encoding bypassed for SpikeYOLO: the encoded tensors are produced beforehand)
    if layer0 is not None:
        with torch.no_grad():
            enc = [orig_forward(x) for x in f32]
        layer0.forward = lambda x: x
        # Check that the bypass really removes only the encoding: the network fed with the pre-encoded tensor
        # (layer 0 = identity) must give the same output as the untouched network fed with the image.
        # Poisson is reproducible here because the eval-mode generator is re-seeded in every call.
        net.eval()
        out_bypass = _first(forward(enc[0]))
        layer0.forward = orig_forward
        out_full = _first(forward(f32[0]))
        layer0.forward = lambda x: x
        scale = float(out_full.float().abs().max()) + 1e-6
        diff = float((out_bypass.float() - out_full.float()).abs().max())
        tol = (5e-2 if a.precision == "amp" else 1e-3) * scale
        bypass_check = dict(max_abs_diff=diff, output_scale=scale, tolerance=tol, passed=diff <= tol)
        print(f"  bypass check: max |bypass - full| = {diff:.2e} (output scale {scale:.2f}, tol {tol:.2e}) -> "
              f"{'OK' if diff <= tol else 'FAILED'}", flush=True)
        if diff > tol:
            layer0.forward = orig_forward
            raise SystemExit("bypass check failed: the 'network' window would not be the full network minus the encoding; "
                             "no numbers were saved.")
        go("network", lambda i: forward(enc[i % P]))
        layer0.forward = orig_forward
        del enc
        go("encode", lambda i: orig_forward(f32[i % P]))
    else:
        go("network", lambda i: forward(f32[i % P]))
    go("endtoend", lambda i: L.decode(forward(pool_u8[i % P].float() / 255)))
    idle_after, _ = run_window("idle_after", None, h, 0, 0, a.batch, idle_seconds=a.idle_seconds)

    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               python=platform.python_version(), host=platform.node(), ultralytics=getattr(L.U, "version", None), ultralytics_file=str(getattr(L.U, "file", None)))
    if h is not None:
        drv = pynvml.nvmlSystemGetDriverVersion()
        env.update(gpu=pynvml.nvmlDeviceGetName(h), driver=drv.decode() if isinstance(drv, bytes) else drv,
                   power_limit_W=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0,
                   sm_clock_max_MHz=pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
    res = dict(model=spec.name, family=spec.family, T=spec.T, encoding=spec.encoding, repeat=a.repeat, batch=a.batch,
               precision=a.precision, params=L.params, fuse=fuse_note, pool=P, warmup=a.warmup, split=a.split,
               peak_mem_GB=(torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None),
               idle_before=idle_before, idle_after=idle_after, windows=windows, env=env, bypass_check=bypass_check,
               codecarbon=dict(enabled=not a.no_codecarbon, country=a.cc_country, error=CC["error"]),
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    np.savez_compressed(out_dir / f"{stem}_latencies.npz", **lat_all)
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"  saved {out_dir / stem}.json")


if __name__ == "__main__":
    main()