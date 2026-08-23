"""COCO-style detection AP with ignore regions.

The matching and ignore logic follows evaluate.py published with
thirdeyelabs/indian-road-benchmark (CC BY 4.0): 101-point interpolated AP over
IoU 0.50:0.95, ground truth marked iscrowd=1 (drawn by only one of two labelling
passes) acts as an ignore region, and optional "near" mode ignores objects under
32 px tall. It is generalised here to arbitrary class sets so all 12 classes can be
scored. The 8-class leaderboard score is computed with the unmodified upstream script.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

THRS = np.arange(0.5, 0.96, 0.05)


def load_gt(path: str | Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    cats = {c["id"]: c["name"] for c in d["categories"]}
    frames = {im["id"]: {**im, "boxes": [], "ignore": []} for im in d["images"]}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        b = [x, y, x + w, y + h, cats[a["category_id"]]]
        frames[a["image_id"]]["ignore" if a.get("iscrowd") else "boxes"].append(b)
    return list(frames.values())


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def ap_from_records(recs: list, npos: int) -> list[float] | None:
    """recs: (score, tp[thr], ignored[thr]). Returns AP per IoU threshold, None when the class has no GT."""
    if npos == 0:
        return None
    recs = sorted(recs, key=lambda r: -r[0])
    out = []
    for t in range(len(THRS)):
        tp = np.array([r[1][t] for r in recs if not r[2][t]], dtype=float)
        if len(tp) == 0:
            out.append(0.0)
            continue
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = ctp / npos
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        prec = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, np.linspace(0, 1, 101), side="left")
        out.append(float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx])))
    return out


def _arr(boxes) -> np.ndarray:
    return np.array([b[:4] for b in boxes], dtype=float).reshape(-1, 4)


def evaluate(frames: list[dict], preds: dict[int, list], classes: list[str],
             gt_map: dict[str, str] | None = None, pred_map: dict[str, str] | None = None,
             floor: float | None = None) -> dict:
    """preds[image_id] = [[x1, y1, x2, y2, score, class_name], ...].

    gt_map / pred_map rename classes into the scored vocabulary; names not in the map are dropped.
    Default is identity over `classes`.
    """
    gt_map = gt_map or {c: c for c in classes}
    pred_map = pred_map or {c: c for c in classes}
    recs = collections.defaultdict(list)
    npos = collections.Counter()
    for f in frames:
        P = preds.get(f["id"], [])
        for c in classes:
            g = [b for b in f["boxes"] if gt_map.get(b[4]) == c]
            ign = [b for b in f["ignore"] if gt_map.get(b[4]) == c]
            if floor:
                ign += [b for b in g if b[3] - b[1] < floor]
                g = [b for b in g if b[3] - b[1] >= floor]
            p = sorted([q for q in P if pred_map.get(q[5]) == c], key=lambda q: -q[4])
            npos[c] += len(g)
            if not p:
                continue
            G, I, Pa = _arr(g), _arr(ign), _arr(p)
            M, MI = iou_matrix(Pa, G), iou_matrix(Pa, I)
            taken = [set() for _ in THRS]
            for k, q in enumerate(p):
                tps, igs = [], []
                for t, thr in enumerate(THRS):
                    best, bj = thr, -1
                    for j in range(len(g)):
                        if j not in taken[t] and M[k, j] >= best:
                            best, bj = M[k, j], j
                    if bj >= 0:
                        taken[t].add(bj)
                        tps.append(1)
                        igs.append(False)
                    else:
                        small = bool(floor) and (q[3] - q[1]) < floor
                        on_ign = len(ign) > 0 and MI[k].max() >= thr
                        tps.append(0)
                        igs.append(bool(small or on_ign))
                recs[c].append((q[4], tps, igs))
    per = {}
    for c in classes:
        a = ap_from_records(recs[c], npos[c])
        per[c] = None if a is None else {"ap50": a[0], "ap": float(np.mean(a)), "n_gt": int(npos[c])}
    valid = [v for v in per.values() if v]
    return {
        "map": float(np.mean([v["ap"] for v in valid])) if valid else 0.0,
        "map50": float(np.mean([v["ap50"] for v in valid])) if valid else 0.0,
        "per_class": per,
    }


def preds_to_coco_results(preds: dict[int, list]) -> list[dict]:
    out = []
    for img, rows in preds.items():
        for x1, y1, x2, y2, s, name in rows:
            out.append({"image_id": img, "category_name": name,
                        "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                        "score": round(float(s), 5)})
    return out


def called_as(frames: list[dict], preds: dict[int, list], cls: str, conf: float = 0.25,
              iou: float = 0.5) -> dict:
    """What the model calls each ground-truth box of `cls`: the class of the highest-IoU prediction
    with score >= conf, or "missed" when that IoU is below `iou`. Ignore regions are not counted."""
    called: collections.Counter = collections.Counter()
    for f in frames:
        g = [b for b in f["boxes"] if b[4] == cls]
        if not g:
            continue
        p = [q for q in preds.get(f["id"], []) if q[4] >= conf]
        M = iou_matrix(_arr(g), _arr(p))
        for i in range(len(g)):
            j = int(M[i].argmax()) if len(p) else -1
            called[p[j][5] if j >= 0 and M[i, j] >= iou else "missed"] += 1
    n = sum(called.values())
    return {"n": n, "called": {k: v / n for k, v in called.most_common()} if n else {}}


def load_coco_results(path: str | Path) -> dict[int, list]:
    """Inverse of preds_to_coco_results."""
    out: dict[int, list] = collections.defaultdict(list)
    for p in json.loads(Path(path).read_text()):
        x, y, w, h = p["bbox"]
        out[p["image_id"]].append([x, y, x + w, y + h, p["score"], p["category_name"]])
    return out
