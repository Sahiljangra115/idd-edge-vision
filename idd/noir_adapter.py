"""Adapter so this ONNX detector can replace NOIR's YOLO `HumanDetector` in Noir/tracker.py.

NOIR's tracker calls `detector.detect(frame) -> List[Detection]` where Detection has
(label: str, bbox: (x1, y1, x2, y2) ints, confidence: float, kind: str). This class returns
objects with that exact shape and needs no import from NOIR (NOIR is not modified).
Use `det_cls=` to pass NOIR's own `Detection` class if you want real instances.

  from idd.noir_adapter import IndianRoadDetector
  det = IndianRoadDetector("exports/int8_minmax.onnx", classes=["person", "rider"])
  dets = det.detect(frame_bgr)

See docs/NOIR.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from .classes import CLASSES
from .infer import OnnxDetector


@dataclass
class Detection:
    label: str
    bbox: tuple[int, int, int, int]
    confidence: float
    kind: str


class IndianRoadDetector:
    def __init__(self, model: str, provider: str = "cpu", conf: float = 0.35, iou: float = 0.5,
                 classes: list[str] | None = None, kind: str = "human", det_cls=Detection):
        self.det = OnnxDetector(model, provider)
        self.conf, self.iou, self.kind, self.det_cls = conf, iou, kind, det_cls
        self.keep = None if classes is None else {CLASSES.index(c) for c in classes}
        self.enabled = True

    def detect(self, frame) -> list:
        if frame is None:
            return []
        out = []
        for x1, y1, x2, y2, s, c in self.det(frame, conf_thr=self.conf, iou_thr=self.iou):
            if self.keep is not None and int(c) not in self.keep:
                continue
            out.append(self.det_cls(CLASSES[int(c)], (int(x1), int(y1), int(x2), int(y2)), float(s), self.kind))
        return out
