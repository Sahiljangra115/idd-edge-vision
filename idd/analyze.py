"""Per-class error analysis for the India-specific classes, baseline vs fine-tuned.

  python -m idd.analyze --runs coco_yolo11n_baseline pt_fp32 --out results/error_analysis.md

Reads results/<run>.json and results/<run>_preds.json from idd.evaluate. For autorickshaw, animal
and rider it reports AP, AP50, GT count, and what each ground-truth box gets called (class of the
best-overlapping box at score >= 0.25, IoU >= 0.5, else "missed"). Writes markdown and a JSON twin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classes import INDIA_CLASSES
from .metrics import called_as, load_coco_results, load_gt

BENCH = Path("data/bench")


def analyze(runs: list[str], rdir: Path, frames: list[dict]) -> dict:
    out = {}
    for run in runs:
        res = json.loads((rdir / f"{run}.json").read_text())
        preds = load_coco_results(rdir / f"{run}_preds.json")
        out[run] = {c: {**(res["per_class"][c] or {}), **called_as(frames, preds, c)} for c in INDIA_CLASSES}
    return out


def to_markdown(a: dict) -> str:
    runs = list(a)
    lines = ["| Class | n GT | " + " | ".join(f"{r} AP / AP50" for r in runs) + " |",
             "|---|---|" + "---|" * len(runs)]
    for c in INDIA_CLASSES:
        n = a[runs[0]][c].get("n_gt", "")
        cells = [f"{100 * a[r][c]['ap']:.1f} / {100 * a[r][c]['ap50']:.1f}" if "ap" in a[r][c] else "n/a" for r in runs]
        lines.append(f"| {c} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "What each ground-truth box is called (score >= 0.25, IoU >= 0.5):", ""]
    for c in INDIA_CLASSES:
        for r in runs:
            called = ", ".join(f"{k} {100 * v:.0f}%" for k, v in a[r][c]["called"].items())
            lines.append(f"- {c}, {r} (n={a[r][c]['n']}): {called}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=["coco_yolo11n_baseline", "pt_fp32"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/error_analysis.md")
    a = ap.parse_args()
    res = analyze(a.runs, Path(a.results), load_gt(BENCH / "dev.json"))
    md = to_markdown(res)
    Path(a.out).write_text(md)
    Path(a.out).with_suffix(".json").write_text(json.dumps(res, indent=2))
    print(md)


if __name__ == "__main__":
    main()
