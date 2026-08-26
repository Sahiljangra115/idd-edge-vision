"""Export a trained .pt to static-shape ONNX at FP32 and FP16.

  python -m idd.export --weights weights/yolo11n_indian_best.pt --out exports

TensorRT engines are built separately from the FP32 ONNX with idd/trt_build.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def export_onnx(weights: Path, out_dir: Path, imgsz: int = 640, half: bool = False) -> Path:
    from ultralytics import YOLO
    produced = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=False,
                                              dynamic=False, half=half, batch=1, device=0 if half else "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{weights.stem}_{'fp16' if half else 'fp32'}.onnx"
    shutil.move(str(produced), dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="exports")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"],
                    help="fp16 export runs on the GPU, fp32 on the CPU")
    a = ap.parse_args()
    for p in a.precisions:
        print(export_onnx(Path(a.weights), Path(a.out), a.imgsz, half=p == "fp16"))


if __name__ == "__main__":
    main()
