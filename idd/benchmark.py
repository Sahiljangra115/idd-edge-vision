"""Latency benchmark for ONNX models on CPU and CUDA with ONNX Runtime.

  python -m idd.benchmark --models exports/a_fp32.onnx exports/a_fp16.onnx exports/a_int8.onnx \
      --providers cpu cuda --runs 200 --out results/latency.json

Measures batch-1, 640x640, single stream. Two numbers per row:
  model   one session.run() on a pre-built input (includes host<->device copies on CUDA)
  e2e     letterbox + session.run() + decode + NMS on a real frame (what a camera loop pays)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from .infer import PROVIDERS, decode, letterbox, make_session


def summarize(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "fps": float(1000.0 / a.mean()), "runs": len(a)}


def bench_one(model: str, provider: str, frames: list[np.ndarray], runs: int, warmup: int,
              threads: int | None, imgsz: int = 640) -> dict:
    sess = make_session(model, provider, threads)
    inp = sess.get_inputs()[0]
    dt = np.float16 if "float16" in inp.type else np.float32
    chw, r, pad = letterbox(frames[0], imgsz)
    x = chw[None].astype(dt)
    for _ in range(warmup):
        sess.run(None, {inp.name: x})
    model_ms = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        model_ms.append((time.perf_counter() - t) * 1e3)
    e2e_ms = []
    for i in range(runs):
        f = frames[i % len(frames)]
        t = time.perf_counter()
        c, r, pad = letterbox(f, imgsz)
        raw = sess.run(None, {inp.name: c[None].astype(dt)})[0]
        decode(raw, r, pad, f.shape[:2], conf_thr=0.25, iou_thr=0.7)
        e2e_ms.append((time.perf_counter() - t) * 1e3)
    return {"model": model, "provider": provider, "active_providers": sess.get_providers(),
            "size_mb": round(os.path.getsize(model) / 1e6, 2),
            "model_only": summarize(model_ms), "e2e": summarize(e2e_ms)}


def machine_info() -> dict:
    import onnxruntime as ort
    info = {"cpu": platform.processor() or platform.machine(), "cores": os.cpu_count(),
            "onnxruntime": ort.__version__, "platform": platform.platform()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import subprocess
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--providers", nargs="+", default=["cpu", "cuda"], choices=PROVIDERS)
    ap.add_argument("--frames", default="data/bench/images/dev")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=None, help="CPU intra-op threads (default: ORT decides)")
    ap.add_argument("--out", default="results/latency.json")
    a = ap.parse_args()

    files = sorted(Path(a.frames).glob("*.jpg"))[:16]
    frames = [cv2.imread(str(p)) for p in files]
    rows = []
    for m in a.models:
        for p in a.providers:
            try:
                row = bench_one(m, p, frames, a.runs, a.warmup, a.threads)
            except Exception as e:  # report, do not hide, e.g. FP16 on CPU or missing CUDA libs
                row = {"model": m, "provider": p, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            rows.append(row)
            if "error" in row:
                print(f"{Path(m).name:34s} {p:5s} ERROR {row['error']}")
            else:
                s, e2e = row["model_only"], row["e2e"]
                print(f"{Path(m).name:34s} {p:5s} model p50 {s['p50_ms']:7.2f} p95 {s['p95_ms']:7.2f} "
                      f"fps {s['fps']:6.1f} | e2e p50 {e2e['p50_ms']:7.2f} p95 {e2e['p95_ms']:7.2f} fps {e2e['fps']:6.1f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"machine": machine_info(), "threads": a.threads, "runs": a.runs,
                                       "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
