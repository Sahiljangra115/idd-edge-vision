"""Build a TensorRT engine from an ONNX model, only if tensorrt is importable.

  python -m idd.trt_build --onnx exports/yolo11n_indian_best_fp32.onnx --out exports/model_fp16.engine --fp16
  python -m idd.trt_build --onnx exports/yolo11n_indian_best_fp32.onnx --out exports/model_int8.engine --int8 --calib data/bench/images/dev

Build the engine ON the target device (engines are not portable across GPUs or TensorRT versions).
On Jetson, trtexec is the simpler route, see docs/TENSORRT_JETSON.md.
INT8 here uses an entropy calibrator over real frames and needs pycuda or cuda-python.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GUIDE = """tensorrt is not importable in this environment, so no engine was built.
Options:
  - Jetson: TensorRT ships with JetPack. Use the system python (python3 -c 'import tensorrt'),
    or run: /usr/src/tensorrt/bin/trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16
  - x86 + NVIDIA GPU: pip install tensorrt (match your CUDA version), then rerun this script.
See docs/TENSORRT_JETSON.md."""


def make_calibrator(trt, calib_dir: str, imgsz: int, cache: str, max_imgs: int = 200):
    import cv2
    import numpy as np
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError:
        raise SystemExit("INT8 calibration needs pycuda (pip install pycuda). FP16 needs nothing extra.")
    from .infer import letterbox

    files = sorted(Path(calib_dir).glob("*.jpg"))[:max_imgs]
    if not files:
        raise SystemExit(f"no .jpg calibration frames in {calib_dir}")

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.buf = cuda.mem_alloc(3 * imgsz * imgsz * 4)

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(files):
                return None
            chw, _, _ = letterbox(cv2.imread(str(files[self.i])), imgsz)
            self.i += 1
            cuda.memcpy_htod(self.buf, np.ascontiguousarray(chw[None]))
            return [int(self.buf)]

        def read_calibration_cache(self):
            p = Path(cache)
            return p.read_bytes() if p.exists() else None

        def write_calibration_cache(self, data):
            Path(cache).write_bytes(data)

    return Calib()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", default="data/bench/images/dev")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    a = ap.parse_args()

    try:
        import tensorrt as trt
    except ImportError:
        print(GUIDE)
        sys.exit(0)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(Path(a.onnx).read_bytes()):
        raise SystemExit("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if a.fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if a.int8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.FP16)  # let non-INT8 layers fall back to FP16
        cfg.int8_calibrator = make_calibrator(trt, a.calib, a.imgsz, str(Path(a.out).with_suffix(".calib")))
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise SystemExit("engine build failed")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(bytes(blob))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
