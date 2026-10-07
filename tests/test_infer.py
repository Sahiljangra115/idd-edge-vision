import numpy as np
import pytest

from idd.infer import decode, letterbox, nms


def test_letterbox_shape_ratio_and_padding():
    img = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
    chw, r, (px, py) = letterbox(img, 640)
    assert chw.shape == (3, 640, 640) and chw.dtype == np.float32
    assert r == 3.2 and px == 0 and py == 160
    assert chw[:, 0, 0].tolist() == pytest.approx([114 / 255] * 3)  # padded rows
    assert chw[:, 320, 320].tolist() == [0.0, 0.0, 0.0]  # image content


def test_nms_suppresses_overlap_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep == [0, 2]


def _raw(boxes_cxcywh, scores_by_class, n_anchors=50, nc=3):
    raw = np.zeros((1, 4 + nc, n_anchors), dtype=np.float32)
    for i, (box, (cls, s)) in enumerate(zip(boxes_cxcywh, scores_by_class)):
        raw[0, :4, i] = box
        raw[0, 4 + cls, i] = s
    return raw


def test_decode_maps_back_to_original_pixels():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    _, r, pad = letterbox(img, 640)
    # original-space box x 20..60, y 10..50 -> letterbox space
    cx, cy, w, h = (40 * r + pad[0], 30 * r + pad[1], 40 * r, 40 * r)
    det = decode(_raw([[cx, cy, w, h]], [(2, 0.8)]), r, pad, (100, 200))
    assert det.shape == (1, 6)
    assert det[0, :4] == pytest.approx([20, 10, 60, 50], abs=1e-3)
    assert det[0, 4] == pytest.approx(0.8) and det[0, 5] == 2


def test_decode_class_aware_nms_and_threshold():
    box = [320, 320, 100, 100]
    raw = _raw([box, box, box], [(0, 0.9), (1, 0.8), (0, 0.0005)])
    det = decode(raw, 1.0, (0, 0), (640, 640), conf_thr=0.001, iou_thr=0.5)
    assert sorted(det[:, 5].tolist()) == [0.0, 1.0]  # same location, different classes both survive
    assert len(decode(raw, 1.0, (0, 0), (640, 640), conf_thr=0.95)) == 0


def test_decode_respects_max_det():
    boxes = [[20 + 12 * i, 50, 10, 10] for i in range(30)]
    det = decode(_raw(boxes, [(0, 0.9)] * 30, n_anchors=40), 1.0, (0, 0), (640, 640), max_det=7)
    assert len(det) == 7
