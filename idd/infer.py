"""Framework-free detector runtime: letterbox, ONNX Runtime session, decode, NMS.

Works on the raw head output of an Ultralytics YOLO detect model exported to ONNX,
shape (1, 4 + num_classes, num_anchors) with boxes as centre-x, centre-y, w, h in
letterboxed pixels. The same code path serves FP32, FP16 and INT8 models, so accuracy
differences between precisions come from the weights and not from the pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

PAD_VALUE = 114


def letterbox(img_bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio, pad to size x size. Returns CHW float32 RGB in [0,1], ratio, (padx, pady)."""
    h, w = img_bgr.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    chw = canvas[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32) / 255.0, r, (float(px), float(py))


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS. Returns kept indices in descending score order."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) <= iou_thr]
    return keep


def decode(raw: np.ndarray, ratio: float, pad: tuple[float, float], orig_hw: tuple[int, int],
           conf_thr: float = 0.001, iou_thr: float = 0.7, max_det: int = 100,
           pre_nms_top: int = 1000) -> np.ndarray:
    """Raw head output (1, 4+nc, N) -> array of [x1, y1, x2, y2, score, cls] in original image pixels."""
    p = raw[0].astype(np.float32).T  # (N, 4+nc)
    cls_scores = p[:, 4:]
    cls = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(p)), cls]
    m = score >= conf_thr
    if not m.any():
        return np.zeros((0, 6), dtype=np.float32)
    p, cls, score = p[m], cls[m], score[m]
    if len(score) > pre_nms_top:
        top = np.argpartition(-score, pre_nms_top)[:pre_nms_top]
        p, cls, score = p[top], cls[top], score[top]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, orig_hw[1])
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, orig_hw[0])
    # class-aware NMS via coordinate offset
    off = cls[:, None].astype(np.float32) * 10000.0
    keep = nms(xyxy + off, score, iou_thr)[:max_det]
    return np.concatenate([xyxy[keep], score[keep, None], cls[keep, None].astype(np.float32)], 1)


PROVIDERS = ["cpu", "cuda", "trt_fp32", "trt_fp16", "trt_int8"]


def _load_tensorrt_libs() -> str:
    """Make libnvinfer.so.10 visible to the ORT TensorRT EP. Looks in ./.pydeps/tensorrt_libs
    (pip --target install) or $IDD_TRT_LIBS. Returns the directory used."""
    import ctypes
    import os
    from pathlib import Path
    cands = [os.environ.get("IDD_TRT_LIBS", ""), str(Path(__file__).resolve().parent.parent / ".pydeps" / "tensorrt_libs")]
    for d in filter(None, cands):
        libs = sorted(Path(d).glob("libnvinfer.so.10*"))
        if libs:
            for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
                f = Path(d) / name
                if f.exists():
                    ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
            return d
    raise RuntimeError("TensorRT 10 libraries not found; set IDD_TRT_LIBS or install tensorrt-cu13-libs into .pydeps")


def make_session(path: str, provider: str = "cpu", threads: int | None = None, trt_cache: str = "exports/trt_cache"):
    """provider: one of PROVIDERS. Raises if CUDA/TensorRT was requested but is not what actually got loaded.

    trt_fp32 / trt_fp16 build a TensorRT engine from the ONNX graph inside ONNX Runtime (TensorRT EP);
    trt_int8 is meant for QDQ models (explicit quantization, scales come from the QDQ nodes).
    Engines are cached under trt_cache so the build cost is paid once per model and precision.
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    cuda = ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE"})
    if provider == "cuda":
        providers = [cuda, "CPUExecutionProvider"]
    elif provider.startswith("trt_"):
        import os
        _load_tensorrt_libs()
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        os.makedirs(trt_cache, exist_ok=True)
        opts = {"trt_engine_cache_enable": True, "trt_engine_cache_path": trt_cache,
                "trt_timing_cache_enable": True, "trt_timing_cache_path": trt_cache,
                "trt_max_workspace_size": 1 << 30,
                "trt_fp16_enable": provider in ("trt_fp16", "trt_int8"),
                "trt_int8_enable": provider == "trt_int8"}
        providers = [("TensorrtExecutionProvider", opts), cuda, "CPUExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        raise ValueError(f"unknown provider {provider!r}, expected one of {PROVIDERS}")
    sess = ort.InferenceSession(path, so, providers=providers)
    want = {"cuda": "CUDAExecutionProvider"}.get(provider, "TensorrtExecutionProvider" if provider.startswith("trt_") else None)
    if want and sess.get_providers()[0] != want:
        raise RuntimeError(f"{provider} requested but session providers are {sess.get_providers()}")
    return sess


class OnnxDetector:
    def __init__(self, path: str, provider: str = "cpu", imgsz: int = 640, threads: int | None = None):
        self.sess = make_session(path, provider, threads)
        self.inp = self.sess.get_inputs()[0]
        self.dtype = np.float16 if "float16" in self.inp.type else np.float32
        self.imgsz = imgsz

    def raw(self, chw: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.inp.name: chw[None].astype(self.dtype)})[0]

    def __call__(self, img_bgr: np.ndarray, **kw) -> np.ndarray:
        chw, r, pad = letterbox(img_bgr, self.imgsz)
        return decode(self.raw(chw), r, pad, img_bgr.shape[:2], **kw)
