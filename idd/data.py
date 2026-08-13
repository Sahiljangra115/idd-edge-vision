"""Dataset preparation.

Training data: WebDataset shards from addu1209/indian-road-dataset (ThirdEye Labs,
CC BY 4.0, real Delhi NCR dashcam frames). Its boxes are MODEL-GENERATED pseudo labels
(each box carries a detector confidence), not human annotation.

Evaluation data: thirdeyelabs/indian-road-benchmark dev split, 450 frames labelled
twice by humans. Clips that appear in the benchmark are removed from training.
"""
from __future__ import annotations

import io
import json
import random
import tarfile
from pathlib import Path

from PIL import Image

from .classes import CLASS_IDX, CLASSES

MAX_SIDE = 960  # frames are 1920x994; store at half size to keep disk and loader cost down


def bdd_labels_to_yolo(labels: list[dict], w: float, h: float, conf_min: float) -> list[str]:
    """BDD100K-style pixel boxes -> YOLO normalised lines, dropping low-confidence and unknown classes."""
    lines = []
    for lab in labels:
        cls = CLASS_IDX.get(lab.get("category"))
        box = lab.get("box2d")
        if cls is None or box is None or lab.get("confidence", 1.0) < conf_min:
            continue
        x1, y1 = max(0.0, box["x1"]), max(0.0, box["y1"])
        x2, y2 = min(float(w), box["x2"]), min(float(h), box["y2"])
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def split_clips(clips: list[str], val_frac: float, seed: int = 0) -> set[str]:
    """Pick validation clips. Splitting by clip stops near-identical neighbouring frames leaking."""
    clips = sorted(set(clips))
    random.Random(seed).shuffle(clips)
    return set(clips[: max(1, round(len(clips) * val_frac))])


def benchmark_clip_ids(dev_json: Path) -> set[str]:
    return {im["clip_id"] for im in json.loads(Path(dev_json).read_text())["images"]}


def _clip(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def extract_shards(shards: list[Path], out: Path, exclude_clips: set[str],
                   conf_min: float = 0.3, val_frac: float = 0.1, seed: int = 0) -> dict:
    """Unpack shards into out/{images,labels}/{train,val}. Returns counts.

    Two passes so memory stays flat: the first lists member names to choose validation clips,
    the second streams each frame to disk.
    """
    clips: set[str] = set()
    for shard in shards:
        with tarfile.open(shard) as tf:
            for m in tf:
                if m.name.endswith(".json") and _clip(m.name[:-5]) not in exclude_clips:
                    clips.add(_clip(m.name[:-5]))
    val_clips = split_clips(sorted(clips), val_frac, seed)
    counts = {"train": 0, "val": 0, "boxes": 0, "excluded_clips": len(exclude_clips)}
    for shard in shards:
        with tarfile.open(shard) as tf:
            jpg: dict[str, bytes] = {}
            for m in tf:
                if "." not in m.name:
                    continue
                stem, ext = m.name.rsplit(".", 1)
                if ext not in ("jpg", "json") or _clip(stem) in exclude_clips:
                    continue
                data = tf.extractfile(m).read()
                if ext == "jpg":
                    jpg[stem] = data
                    continue
                if stem not in jpg:
                    continue
                split = "val" if _clip(stem) in val_clips else "train"
                img = Image.open(io.BytesIO(jpg.pop(stem))).convert("RGB")
                w, h = img.size
                lines = bdd_labels_to_yolo(json.loads(data).get("labels", []), w, h, conf_min)
                if max(w, h) > MAX_SIDE:
                    s = MAX_SIDE / max(w, h)
                    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
                (out / "images" / split).mkdir(parents=True, exist_ok=True)
                (out / "labels" / split).mkdir(parents=True, exist_ok=True)
                img.save(out / "images" / split / f"{stem}.jpg", quality=92)
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
                counts[split] += 1
                counts["boxes"] += len(lines)
    return counts


def write_data_yaml(root: Path) -> Path:
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    p = root / "data.yaml"
    p.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return p


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--dev-json", default="data/bench/dev.json")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--conf-min", type=float, default=0.3)
    a = ap.parse_args()
    out = Path(a.out)
    counts = extract_shards([Path(s) for s in a.shards], out, benchmark_clip_ids(Path(a.dev_json)), a.conf_min)
    write_data_yaml(out)
    (out / "prep_summary.json").write_text(json.dumps(counts, indent=2))
    print(counts)


if __name__ == "__main__":
    main()
