"""Post-training static INT8 quantization (QDQ) with ONNX Runtime and a real-image calibration set.

  python -m idd.quantize --model exports/x_fp32.onnx --calib data/yolo/images/train --n 128 \
      --out exports/x_int8.onnx [--method minmax|percentile|entropy] [--keep-head-fp32]

The detect-head decode tail (DFL, anchor math, class Sigmoid, final Concat) is always left FP32;
see head_nodes() for why quantizing it gives mAP 0. --keep-head-fp32 also leaves the head conv
branches (cv2 box, cv3 class) unquantized.
"""
from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .infer import letterbox


class ImageCalibrationReader:
    """Feeds letterboxed real frames to the ORT calibrator."""

    def __init__(self, input_name: str, images: list[Path], imgsz: int = 640):
        self.name = input_name
        self.paths = iter(images)
        self.imgsz = imgsz

    def get_next(self):
        p = next(self.paths, None)
        if p is None:
            return None
        chw, _, _ = letterbox(cv2.imread(str(p)), self.imgsz)
        return {self.name: chw[None]}

    def rewind(self):
        pass


def pick_calibration_images(folder: Path, n: int, seed: int = 0) -> list[Path]:
    files = sorted(folder.glob("*.jpg"))
    random.Random(seed).shuffle(files)
    return files[:n]


def head_nodes(model_path: Path) -> tuple[list[str], list[str]]:
    """Split the detect head (last model.N block) into (conv-branch nodes, decode nodes).

    Decode nodes are everything in the head outside the cv2/cv3 conv branches: DFL softmax,
    anchor arithmetic, the class Sigmoid and the final Concat. They must stay FP32. Their
    output joins box coordinates (0..640 px) and class scores (0..1) in one tensor; one INT8
    scale for that tensor is about 2.5 px per step, so every class score rounds to 0 and
    mAP is 0 at any precision of the backbone.
    """
    import onnx
    m = onnx.load(str(model_path))
    nodes = [n for n in m.graph.node if n.op_type != "Constant"]
    last = max(int(n.name.split("/model.")[1].split("/")[0]) for n in nodes if "/model." in n.name)
    head = [n for n in nodes if f"/model.{last}/" in n.name]
    branch = [n.name for n in head if f"/model.{last}/cv2." in n.name or f"/model.{last}/cv3." in n.name]
    decode = [n.name for n in head if n.name not in set(branch)]
    return branch, decode


def quantize(model: Path, calib_dir: Path, out: Path, n: int = 128, method: str = "minmax",
             keep_head_fp32: bool = False, imgsz: int = 640) -> Path:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
               "entropy": CalibrationMethod.Entropy}
    branch, decode = head_nodes(model)
    # The decode tail is always FP32. --keep-head-fp32 also keeps the head conv branches FP32.
    excluded = decode + (branch if keep_head_fp32 else [])
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(model), str(pre), skip_optimization=True)
        name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
        reader = ImageCalibrationReader(name, pick_calibration_images(calib_dir, n), imgsz)
        out.parent.mkdir(parents=True, exist_ok=True)
        quantize_static(
            str(pre), str(out), reader,
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            calibrate_method=methods[method],
            nodes_to_exclude=excluded,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default="data/yolo/images/train")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--keep-head-fp32", action="store_true")
    a = ap.parse_args()
    print(quantize(Path(a.model), Path(a.calib), Path(a.out), a.n, a.method, a.keep_head_fp32))


if __name__ == "__main__":
    main()
