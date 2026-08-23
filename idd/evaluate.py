"""Score a detector on the human-labelled dev split of the Indian Road Detection Benchmark.

  python -m idd.evaluate --model weights/best.pt --name pt_fp32
  python -m idd.evaluate --model exports/best_fp32.onnx --provider cuda --name onnx_fp32_cuda
  python -m idd.evaluate --model weights/yolo11n.pt --coco --name coco_baseline

Writes results/<name>.json (12-class metrics, the 8-class upstream leaderboard score,
per-class AP) and results/<name>_preds.json (COCO results format).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2

from .classes import CLASSES, COCO80_TO_OURS, INDIA_CLASSES
from .infer import PROVIDERS, OnnxDetector
from .metrics import evaluate, load_gt, preds_to_coco_results

BENCH = Path("data/bench")
CONDITIONS = {"day": lambda f: f["time_of_day"] == "day", "dusk": lambda f: f["time_of_day"] == "dusk",
              "night": lambda f: f["time_of_day"] == "night", "rain": lambda f: f["rain"]}


def predict_all(model_path: str, frames: list[dict], provider: str, coco: bool, imgsz: int,
                threads: int | None = None) -> dict[int, list]:
    """Return {image_id: [[x1,y1,x2,y2,score,class_name], ...]}, up to 100 boxes per frame."""
    preds: dict[int, list] = {}
    if model_path.endswith(".onnx"):
        det = OnnxDetector(model_path, provider, imgsz, threads)
        names = CLASSES

        def run(img):
            return det(img, conf_thr=0.001, iou_thr=0.7, max_det=100)
    else:
        from ultralytics import YOLO
        yolo = YOLO(model_path)
        names = None

        def run(img):
            r = yolo.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, max_det=100, verbose=False,
                             device=0 if provider == "cuda" else "cpu")[0]
            b = r.boxes
            return [[*xy, s, c] for xy, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]

    for f in frames:
        img = cv2.imread(str(BENCH / "images" / "dev" / f["file_name"]))
        rows = []
        for x1, y1, x2, y2, s, c in run(img):
            c = int(c)
            if coco:
                name = COCO80_TO_OURS.get(c)
            else:
                name = (names or CLASSES)[c]
            if name:
                rows.append([float(x1), float(y1), float(x2), float(y2), float(s), name])
        preds[f["id"]] = rows
    return preds


def upstream_score(gt_path: Path, pred_path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("upstream_eval", BENCH / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score(str(gt_path), str(pred_path))


def score_preds(frames: list[dict], preds: dict[int, list], pred_path: Path | None = None) -> dict:
    res = evaluate(frames, preds, CLASSES)
    res["by_condition"] = {c: evaluate([f for f in frames if t(f)], preds, CLASSES)["map"]
                           for c, t in CONDITIONS.items()}
    res["india_classes"] = {c: res["per_class"][c] for c in INDIA_CLASSES}
    if pred_path is not None:
        up = upstream_score(BENCH / "dev.json", pred_path)
        res["leaderboard_8class"] = {"map": up["map"], "map50": up["map50"], "near": up["near"],
                                     "by_condition": up["by_condition"], "autos": up["autos"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--provider", default="cpu", choices=PROVIDERS)
    ap.add_argument("--coco", action="store_true", help="COCO-pretrained model: map COCO ids to our classes")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N frames only")
    ap.add_argument("--threads", type=int, default=4, help="ONNX Runtime CPU intra-op threads")
    a = ap.parse_args()

    frames = load_gt(BENCH / "dev.json")
    if a.limit:
        frames = frames[: a.limit]
    t0 = time.time()
    preds = predict_all(a.model, frames, a.provider, a.coco, a.imgsz, a.threads)
    out = Path("results")
    out.mkdir(exist_ok=True)
    pred_path = out / f"{a.name}_preds.json"
    pred_path.write_text(json.dumps(preds_to_coco_results(preds)))
    res = score_preds(frames, preds, None if a.limit else pred_path)
    res.update(model=a.model, provider=a.provider, frames=len(frames), eval_seconds=round(time.time() - t0, 1))
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=2))
    lb = res.get("leaderboard_8class")
    print(f"{a.name}: 12-class mAP50-95 {100 * res['map']:.1f}  mAP50 {100 * res['map50']:.1f}"
          + (f"  | 8-class leaderboard score {100 * lb['map']:.1f} (AP50 {100 * lb['map50']:.1f})" if lb else ""))
    for c in CLASSES:
        v = res["per_class"][c]
        print(f"  {c:18s} AP {100 * v['ap']:5.1f}  AP50 {100 * v['ap50']:5.1f}  n_gt {v['n_gt']}" if v else f"  {c:18s} no GT")


if __name__ == "__main__":
    main()
