"""Write README-ready markdown tables from result JSONs.

  python -m idd.report                      # prints to stdout
  python -m idd.report --out results/tables.md

Inputs: results/latency*.json (idd.benchmark) and results/*.json from idd.evaluate.
Evaluation and latency rows are joined on (model path, provider); when a model was only evaluated
on the other provider, that accuracy is shown with a "*". Missing values print "pending".
--prefix selects one run family (e.g. "smoke_"); the default skips smoke_ files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classes import CLASSES

P = "pending"


def mine(name: str, prefix: str) -> bool:
    return name.startswith(prefix) if prefix else not name.startswith("smoke_")


def load_evals(rdir: Path, prefix: str = "") -> list[dict]:
    out = []
    for f in sorted(rdir.glob("*.json")):
        if f.name.endswith("_preds.json") or "latency" in f.name or not mine(f.name, prefix):
            continue
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "per_class" in d:
            d["_name"] = f.stem
            out.append(d)
    return out


def pct(v) -> str:
    return P if v is None else f"{100 * v:.1f}"


def num(v, fmt="{:.2f}") -> str:
    return P if v is None else fmt.format(v)


def accuracy_table(evals: list[dict]) -> str:
    lines = ["| Run | Model | Provider | Frames | mAP50-95 | mAP50 | 8-class score |", "|---|---|---|---|---|---|---|"]
    for e in evals:
        lb = e.get("leaderboard_8class") or {}
        lines.append(f"| {e['_name']} | {Path(e.get('model', '?')).name} | {e.get('provider', '?')} | {e.get('frames', P)} "
                     f"| {pct(e.get('map'))} | {pct(e.get('map50'))} | {pct(lb.get('map'))} |")
    if not evals:
        lines.append(f"| {P} | | | | | | |")
    return "\n".join(lines)


def per_class_table(evals: list[dict]) -> str:
    head = "| Class | n GT | " + " | ".join(e["_name"] for e in evals) + " |"
    lines = [head, "|---|---|" + "---|" * len(evals)]
    for c in CLASSES:
        n = next((e["per_class"][c]["n_gt"] for e in evals if e["per_class"].get(c)), "")
        cells = [pct(e["per_class"][c]["ap"]) if e["per_class"].get(c) else "n/a" for e in evals]
        lines.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latency_table(lat: dict | None, evals: list[dict]) -> str:
    lines = ["| Model | Provider | mAP50-95 | mAP50 | model p50 / p95 ms | FPS | e2e p95 ms | Size MB |",
             "|---|---|---|---|---|---|---|---|"]
    rows = (lat or {}).get("rows", [])
    if not rows:
        lines.append(f"| {P} | | {P} | {P} | {P} | {P} | {P} | {P} |")
    ev = {(e.get("model"), e.get("provider")): e for e in evals}
    any_prov = {e.get("model"): e for e in evals}
    for r in rows:
        name = Path(r["model"]).name
        if "error" in r:
            lines.append(f"| {name} | {r['provider']} | error: {r['error'][:60]} | | | | | |")
            continue
        e = ev.get((r["model"], r["provider"]))
        star = "" if e else "*"
        e = e or any_prov.get(r["model"], {})
        m, e2 = r["model_only"], r["e2e"]
        acc = f"{pct(e.get('map'))}{star if e else ''} | {pct(e.get('map50'))}{star if e else ''}"
        lines.append(f"| {name} | {r['provider']} | {acc} "
                     f"| {m['p50_ms']:.2f} / {m['p95_ms']:.2f} | {m['fps']:.1f} | {e2['p95_ms']:.2f} | {r['size_mb']} |")
    if lat:
        mi = lat.get("machine", {})
        lines.append(f"\nMachine: {mi.get('cpu', '?')}, {mi.get('gpu', 'no GPU')}, onnxruntime {mi.get('onnxruntime', '?')}, "
                     f"{lat.get('runs', '?')} runs, batch 1, 640x640, CPU threads {lat.get('threads') or 'ORT default'}. "
                     "* = accuracy measured on the other provider.")
    return "\n".join(lines)


def load_latency(rdir: Path, prefix: str = "") -> dict | None:
    """Merge results/<prefix>latency*.json files into one {machine, threads, runs, rows}."""
    files = [f for f in sorted(rdir.glob(f"{prefix}latency*.json")) if mine(f.name, prefix)]
    if not files:
        return None
    parts = [json.loads(f.read_text()) for f in files]
    merged = {**parts[0], "rows": [r for p in parts for r in p["rows"]]}
    merged["threads"] = next((p["threads"] for p in parts if p.get("threads")), None)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    rdir = Path(a.results)
    evals = load_evals(rdir, a.prefix)
    lat = load_latency(rdir, a.prefix)
    md = ("### Accuracy\n\n" + accuracy_table(evals) + "\n\n### Per-class AP50-95\n\n" + per_class_table(evals)
          + "\n\n### Latency per precision and device\n\n" + latency_table(lat, evals) + "\n")
    if a.out:
        Path(a.out).write_text(md)
    else:
        print(md)


if __name__ == "__main__":
    main()
