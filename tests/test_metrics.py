import pytest

from idd.metrics import evaluate, iou_matrix, preds_to_coco_results

import numpy as np

CLASSES = ["car", "autorickshaw"]


def frame(boxes=(), ignore=(), fid=1):
    return {"id": fid, "boxes": list(boxes), "ignore": list(ignore)}


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10.0]])
    b = np.array([[0, 0, 10, 10.0], [5, 0, 15, 10.0], [20, 20, 30, 30.0]])
    assert iou_matrix(a, b)[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_perfect_predictions_give_ap_one():
    f = frame([[0, 0, 100, 100, "car"], [200, 200, 300, 300, "autorickshaw"]])
    preds = {1: [[0, 0, 100, 100, 0.9, "car"], [200, 200, 300, 300, 0.9, "autorickshaw"]]}
    r = evaluate([f], preds, CLASSES)
    assert r["map"] == pytest.approx(1.0) and r["map50"] == pytest.approx(1.0)


def test_no_predictions_give_zero_and_missing_class_is_skipped():
    f = frame([[0, 0, 100, 100, "car"]])
    r = evaluate([f], {}, CLASSES)
    assert r["map"] == 0.0
    assert r["per_class"]["autorickshaw"] is None  # no GT: excluded from the mean, not counted as zero


def test_false_positive_lowers_ap_but_ignore_region_does_not():
    gt = [[0, 0, 100, 100, "car"]]
    good = [0, 0, 100, 100, 0.9, "car"]
    fp = [500, 500, 600, 600, 0.95, "car"]  # higher score than the true positive
    base = evaluate([frame(gt)], {1: [good]}, CLASSES)["map"]
    with_fp = evaluate([frame(gt)], {1: [good, fp]}, CLASSES)["map"]
    ignored = evaluate([frame(gt, ignore=[[500, 500, 600, 600, "car"]])], {1: [good, fp]}, CLASSES)["map"]
    assert with_fp < base
    assert ignored == pytest.approx(base)


def test_looser_localisation_hurts_map_more_than_map50():
    f = frame([[0, 0, 100, 100, "car"]])
    r = evaluate([f], {1: [[0, 0, 100, 70, 0.9, "car"]]}, CLASSES)  # IoU 0.7
    assert r["map50"] == pytest.approx(1.0) and r["map"] < 0.7


def test_near_floor_ignores_small_objects():
    gt = frame([[0, 0, 10, 10, "car"], [100, 100, 150, 200, "car"]])
    assert evaluate([gt], {}, CLASSES)["per_class"]["car"]["n_gt"] == 2
    assert evaluate([gt], {}, CLASSES, floor=32)["per_class"]["car"]["n_gt"] == 1


def test_class_maps_merge_rider_into_person():
    f = frame([[0, 0, 50, 100, "rider"]])
    r = evaluate([f], {1: [[0, 0, 50, 100, 0.9, "person"]]}, ["person"], gt_map={"rider": "person"},
                 pred_map={"person": "person"})
    assert r["map"] == pytest.approx(1.0)


def test_coco_results_format():
    out = preds_to_coco_results({7: [[10, 20, 40, 60, 0.5, "car"]]})
    assert out == [{"image_id": 7, "category_name": "car", "bbox": [10, 20, 30, 40], "score": 0.5}]


def test_called_as_reports_confusion_and_misses():
    from idd.metrics import called_as
    f = frame([[0, 0, 100, 100, "autorickshaw"], [300, 300, 400, 400, "autorickshaw"]])
    preds = {1: [[0, 0, 100, 95, 0.9, "car"], [300, 300, 400, 400, 0.1, "autorickshaw"]]}
    r = called_as([f], preds, "autorickshaw", conf=0.25)
    assert r["n"] == 2 and r["called"] == {"car": 0.5, "missed": 0.5}


def test_coco_results_round_trip(tmp_path):
    import json
    from idd.metrics import load_coco_results
    rows = {3: [[10.0, 20.0, 40.0, 60.0, 0.5, "car"]]}
    (tmp_path / "p.json").write_text(json.dumps(preds_to_coco_results(rows)))
    assert load_coco_results(tmp_path / "p.json")[3] == rows[3]
