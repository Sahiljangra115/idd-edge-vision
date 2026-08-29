"""Video / webcam demo with the ONNX Runtime detector. Boxes, labels and an FPS overlay.

  python -m idd.demo --model exports/yolo11n_indian_best_fp32.onnx --source 0              # webcam
  python -m idd.demo --model exports/int8_minmax.onnx --source clip.mp4 --out demo.mp4 --no-show
  python -m idd.demo --model exports/int8_minmax.onnx --source "frames/clip_*.jpg" --fps 5 --out demo.mp4 --no-show --snapshots 3
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import cv2

from .classes import CLASSES
from .infer import PROVIDERS, OnnxDetector


def color(c: int) -> tuple[int, int, int]:
    return (37 * c % 256, 97 * (c + 3) % 256, 173 * (c + 5) % 256)


def draw(img, dets) -> None:
    for x1, y1, x2, y2, s, c in dets:
        c = int(c)
        col = color(c)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"{CLASSES[c]} {s:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


class Frames:
    """cv2.VideoCapture-like reader over a sorted glob of images, so a frame sequence works as a source."""

    def __init__(self, pattern: str, fps: float):
        self.files = sorted(glob.glob(pattern))
        self.i, self.fps = 0, fps

    def isOpened(self) -> bool:
        return bool(self.files)

    def read(self):
        if self.i >= len(self.files):
            return False, None
        self.i += 1
        return True, cv2.imread(self.files[self.i - 1])

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else len(self.files)

    def release(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="0", help="camera index, video file, or quoted image glob")
    ap.add_argument("--fps", type=float, default=10.0, help="output frame rate for an image-glob source")
    ap.add_argument("--snapshots", type=int, default=0, help="also save this many evenly spaced PNG frames next to --out")
    ap.add_argument("--provider", default="cpu", choices=PROVIDERS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=None, help="write annotated video here")
    ap.add_argument("--no-show", action="store_true", help="headless: no window")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    det = OnnxDetector(a.model, a.provider, a.imgsz)
    if any(ch in a.source for ch in "*?["):
        cap = Frames(a.source, a.fps)
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.source}")
    writer = None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if a.max_frames:
        total = min(total, a.max_frames) if total else a.max_frames
    snap_at = {round((k + 0.5) * total / a.snapshots) for k in range(a.snapshots)} if a.snapshots and total else set()
    fps_s, n, prev = 0.0, 0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        dets = det(frame, conf_thr=a.conf, iou_thr=a.iou)
        now = time.perf_counter()
        inst = 1.0 / max(now - prev, 1e-9)
        prev = now
        fps_s = inst if n == 0 else 0.9 * fps_s + 0.1 * inst  # smoothed, includes capture + draw
        draw(frame, dets)
        cv2.putText(frame, f"{fps_s:5.1f} FPS  {len(dets)} det  {a.provider}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if a.out:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 30, (w, h))
            writer.write(frame)
            if n in snap_at:
                png = Path(a.out).with_name(f"{Path(a.out).stem}_{n:04d}.png")
                cv2.imwrite(str(png), frame)
                print("saved", png)
        if not a.no_show:
            cv2.imshow("idd-edge-vision", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        n += 1
    cap.release()
    if writer:
        writer.release()
    if not a.no_show:
        cv2.destroyAllWindows()
    print(f"{n} frames, last smoothed FPS {fps_s:.1f}")


if __name__ == "__main__":
    main()
