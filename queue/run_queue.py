#!/usr/bin/env python3
"""Run a list of training jobs one after another, unattended.

* Jobs are defined in a Python file (default: jobs.py) that exposes JOBS = [dict, ...].
* Each job runs as a subprocess; its output goes to logs/<name>.log.
* queue_state.json records done / failed / running for every job, so the queue can be
  stopped and started again: finished jobs are skipped, an interrupted job is resumed
  (if the job defines resume_args and its checkpoint exists) or archived and restarted.
* A failing job does not stop the queue; the next job starts.
* Create a file called STOP in the working directory to stop after the current job.

Usage:
    python run_queue.py --dry-run          # print what would run
    nohup python run_queue.py > queue.out 2>&1 &     (or inside tmux)
    python run_queue.py --retry-failed     # run again the jobs that failed
    python run_queue.py --only spikeyolo_rgb_T4_s0
"""
import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time

STATE_FILE = "queue_state.json"
LOG_DIR = "logs"


def load_jobs(path):
    spec = importlib.util.spec_from_file_location("jobs_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jobs = mod.JOBS
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate job names in " + path)
    return jobs


def load_state(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="jobs.py")
    ap.add_argument("--state", default=STATE_FILE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these job names")
    ap.add_argument("--min-free-gb", type=float, default=20.0, help="stop if free disk is below this")
    a = ap.parse_args()

    jobs = load_jobs(a.jobs)
    state = load_state(a.state)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---- plan ----
    plan = []
    for j in jobs:
        name = j["name"]
        st = state.get(name, {}).get("status")
        if a.only and name not in a.only:
            continue
        if not j.get("enabled", True):
            print(f"[skip]    {name}: disabled")
            continue
        if st == "done":
            print(f"[skip]    {name}: already done")
            continue
        if st == "failed" and not a.retry_failed:
            print(f"[skip]    {name}: failed before (use --retry-failed)")
            continue
        plan.append(j)

    print(f"\n{len(plan)} job(s) to run:")
    for j in plan:
        print("  -", j["name"], ":", " ".join(j["cmd"]))
    if a.dry_run or not plan:
        return

    stop = {"flag": False, "proc": None}

    def on_signal(signum, frame):
        stop["flag"] = True
        p = stop["proc"]
        if p and p.poll() is None:
            print(f"\n[signal {signum}] stopping the running job (it can be resumed later)")
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    for j in plan:
        if stop["flag"] or os.path.exists("STOP"):
            print("[stop] STOP requested; leaving the rest of the queue untouched")
            break
        free = shutil.disk_usage(".").free / 1e9
        if free < a.min_free_gb:
            print(f"[stop] only {free:.1f} GB free (< {a.min_free_gb}); stopping the queue")
            break

        name = j["name"]
        cmd = list(j["cmd"])
        prev = state.get(name, {})
        # interrupted before? resume it if possible, otherwise archive the partial run
        if prev.get("status") == "running":
            ckpt = j.get("resume_check")
            if j.get("resume_args") and ckpt and os.path.isfile(ckpt):
                cmd += j["resume_args"]
                print(f"[resume]  {name}: continuing from {ckpt}")
            elif j.get("run_dir") and os.path.isdir(j["run_dir"]):
                new = j["run_dir"].rstrip("/") + "_interrupted_" + time.strftime("%Y%m%d_%H%M%S")
                os.rename(j["run_dir"], new)
                print(f"[restart] {name}: partial run moved to {new}")

        log_path = os.path.join(LOG_DIR, name + ".log")
        state[name] = {"status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd}
        save_state(a.state, state)
        print(f"[start]   {name}  ({time.strftime('%H:%M:%S')})  log: {log_path}")

        t0 = time.time()
        env = os.environ.copy()
        env.update(j.get("env", {}))
        with open(log_path, "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=j.get("cwd", "."), stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            stop["proc"] = proc
            rc = proc.wait()
            stop["proc"] = None
        dt = time.time() - t0

        if stop["flag"]:
            state[name].update(status="running", note="interrupted by signal")  # resumable
            save_state(a.state, state)
            print(f"[paused]  {name} after {fmt(dt)}")
            break
        done_file = j.get("done_file")
        ok = rc == 0 and (done_file is None or os.path.isfile(done_file))
        state[name].update(status="done" if ok else "failed", returncode=rc, seconds=round(dt),
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_state(a.state, state)
        print(f"[{'done' if ok else 'FAILED'}]  {name}  rc={rc}  {fmt(dt)}")

    print("\n===== summary =====")
    for j in jobs:
        s = state.get(j["name"], {})
        print(f"{j['name']:<40} {s.get('status', '-'):<8} {fmt(s['seconds']) if 'seconds' in s else ''}")


if __name__ == "__main__":
    main()