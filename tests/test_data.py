import io
import json
import tarfile

from PIL import Image

from idd.classes import CLASS_IDX
from idd.data import bdd_labels_to_yolo, extract_shards, split_clips


def lab(cat, x1, y1, x2, y2, conf=0.9):
    return {"category": cat, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "confidence": conf}


def test_yolo_conversion_normalises_and_maps_class():
    out = bdd_labels_to_yolo([lab("autorickshaw", 100, 200, 300, 400)], 1000, 1000, 0.3)
    cls, cx, cy, w, h = out[0].split()
    assert int(cls) == CLASS_IDX["autorickshaw"]
    assert (float(cx), float(cy), float(w), float(h)) == (0.2, 0.3, 0.2, 0.2)


def test_low_confidence_unknown_class_and_tiny_boxes_are_dropped():
    labels = [lab("car", 0, 0, 50, 50, conf=0.1), lab("spaceship", 0, 0, 50, 50), lab("car", 10, 10, 11, 11)]
    assert bdd_labels_to_yolo(labels, 100, 100, 0.3) == []


def test_boxes_are_clipped_to_image():
    out = bdd_labels_to_yolo([lab("car", -20, 50, 120, 80)], 100, 100, 0.3)
    _, cx, _, w, _ = map(float, out[0].split())
    assert w == 1.0 and cx == 0.5


def test_clip_split_is_deterministic_and_never_empty():
    clips = [f"c{i}" for i in range(20)]
    a, b = split_clips(clips, 0.1), split_clips(list(reversed(clips)), 0.1)
    assert a == b and len(a) == 2
    assert len(split_clips(["only"], 0.1)) == 1


def _jpg(size=(400, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _add(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_extract_shards_excludes_benchmark_clips_and_splits_by_clip(tmp_path):
    shard = tmp_path / "s.tar"
    with tarfile.open(shard, "w") as tf:
        for clip in ["keep1", "keep2", "bench1"]:
            for f in range(2):
                stem = f"{clip}_{f:04d}"
                _add(tf, f"{stem}.jpg", _jpg())
                _add(tf, f"{stem}.png", b"mask")
                _add(tf, f"{stem}.json", json.dumps({"labels": [lab("car", 10, 10, 100, 100)]}).encode())
    out = tmp_path / "yolo"
    counts = extract_shards([shard], out, {"bench1"}, conf_min=0.3, val_frac=0.5)
    assert counts["train"] + counts["val"] == 4  # bench1 removed
    assert not list(out.rglob("bench1*"))
    train_clips = {p.stem.rsplit("_", 1)[0] for p in (out / "images" / "train").glob("*.jpg")}
    val_clips = {p.stem.rsplit("_", 1)[0] for p in (out / "images" / "val").glob("*.jpg")}
    assert train_clips.isdisjoint(val_clips)
    assert counts["boxes"] == 4
