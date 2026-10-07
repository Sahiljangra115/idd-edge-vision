"""Session wiring and benchmark plumbing, using a tiny synthetic ONNX model (no trained weights needed)."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

from idd.benchmark import bench_one, summarize  # noqa: E402
from idd.infer import OnnxDetector  # noqa: E402
from idd.quantize import head_nodes, pick_calibration_images  # noqa: E402


def tiny_model(path, nc=3, anchors=20):
    """Input (1,3,640,640) -> constant head (1, 4+nc, anchors) with one confident box at the image centre."""
    head = np.zeros((1, 4 + nc, anchors), dtype=np.float32)
    head[0, :4, 0] = [320, 320, 64, 64]
    head[0, 4 + 1, 0] = 0.9
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4 + nc, anchors])
    const = helper.make_tensor("h", TensorProto.FLOAT, head.shape, head.flatten().tolist())
    # Add(ReduceMean(x) * 0, h) keeps a real data dependency on the input
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], keepdims=0),
        helper.make_node("Constant", [], ["z"], value=helper.make_tensor("zv", TensorProto.FLOAT, [], [0.0])),
        helper.make_node("Mul", ["m", "z"], ["mz"]),
        helper.make_node("Constant", [], ["hc"], value=const),
        helper.make_node("Add", ["hc", "mz"], ["output0"]),
    ]
    g = helper.make_graph(nodes, "tiny", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save(m, str(path))
    return path


def test_detector_end_to_end_on_cpu(tmp_path):
    det = OnnxDetector(str(tiny_model(tmp_path / "t.onnx")), "cpu", 640)
    img = np.zeros((497, 960, 3), dtype=np.uint8)
    out = det(img, conf_thr=0.25)
    assert out.shape == (1, 6) and out[0, 5] == 1
    cx, cy = (out[0, 0] + out[0, 2]) / 2, (out[0, 1] + out[0, 3]) / 2
    assert (cx, cy) == pytest.approx((480, 248.5), abs=1)  # centre of the letterboxed canvas maps to image centre


def test_cuda_request_fails_loudly_instead_of_silently_using_cpu(tmp_path):
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA available here")
    with pytest.raises(Exception):
        OnnxDetector(str(tiny_model(tmp_path / "t.onnx")), "cuda")


def test_bench_one_reports_ordered_percentiles(tmp_path):
    frames = [np.zeros((497, 960, 3), dtype=np.uint8)]
    row = bench_one(str(tiny_model(tmp_path / "t.onnx")), "cpu", frames, runs=15, warmup=2, threads=1)
    s = row["model_only"]
    assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] and s["fps"] > 0 and row["e2e"]["runs"] == 15


def test_summarize_fps_is_inverse_mean():
    s = summarize([10.0, 10.0, 10.0])
    assert s["fps"] == pytest.approx(100.0) and s["p95_ms"] == 10.0


def test_calibration_image_pick_is_deterministic(tmp_path):
    for i in range(10):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
    a = pick_calibration_images(tmp_path, 4)
    assert a == pick_calibration_images(tmp_path, 4) and len(a) == 4


def test_head_split_keeps_decode_tail_out_of_int8(tmp_path):
    """Regression: quantizing the final Concat (boxes 0..640 with scores 0..1) zeroed every score."""
    x = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("output0", TensorProto.FLOAT, None)
    w = helper.make_tensor("w", TensorProto.FLOAT, [3, 3, 1, 1], [0.1] * 9)
    nodes = [
        helper.make_node("Conv", ["images", "w"], ["a"], name="/model.0/conv/Conv"),
        helper.make_node("Conv", ["a", "w"], ["b"], name="/model.23/cv2.0/cv2.0.2/Conv"),
        helper.make_node("Conv", ["a", "w"], ["c"], name="/model.23/cv3.0/cv3.0.2/Conv"),
        helper.make_node("Sigmoid", ["c"], ["s"], name="/model.23/Sigmoid"),
        helper.make_node("Concat", ["b", "s"], ["output0"], axis=1, name="/model.23/Concat_3"),
    ]
    m = helper.make_model(helper.make_graph(nodes, "h", [x], [y], [w]), opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(m, str(tmp_path / "h.onnx"))
    branch, decode = head_nodes(tmp_path / "h.onnx")
    assert set(branch) == {"/model.23/cv2.0/cv2.0.2/Conv", "/model.23/cv3.0/cv3.0.2/Conv"}
    assert set(decode) == {"/model.23/Sigmoid", "/model.23/Concat_3"}
