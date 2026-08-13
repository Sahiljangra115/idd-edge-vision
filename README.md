# Indian road scene detection on the edge

Fine-tune a small YOLO detector on real Indian dashcam frames, export to ONNX, quantize,
and measure accuracy and latency per precision on CPU and CUDA with ONNX Runtime.
The point is the edge table and the per-class error analysis, not the YOLO fine-tune.

Status: code, data pipeline and COCO baseline are done. Fine-tuning, export, quantization,
accuracy-per-precision and latency are **pending, run later** (`run_all.sh`).

## Dataset: what is and is not IDD

The India Driving Dataset (IIIT Hyderabad) needs a registered account at
idd.insaan.iiit.ac.in. I did not use it. No account-free official subset is reachable.
Unofficial Hugging Face mirrors of IDD exist (for example `sanak/IDD`, a 24 GB tarball
with no licence statement and no provenance), and I did not use them either.

Used instead, both open, CC BY 4.0, real Delhi NCR dashcam footage, no login:

| Role | Dataset | Labels |
|---|---|---|
| Train | `addu1209/indian-road-dataset` (ThirdEye Labs, 646 shards; 4 used: indices 10, 150, 300, 450) | **Pseudo labels.** Each box carries a detector confidence; the card says boxes come from a custom YOLO. Kept at confidence >= 0.3. |
| Evaluate | `thirdeyelabs/indian-road-benchmark`, dev split, 450 frames, 309 clips | **Human.** Two independent annotation passes; agreed boxes are ground truth, single-pass boxes are ignore regions. |

Consequences, stated plainly:

- Results are **not comparable to the NCVPRIPG IDD leaderboard** and use no IDD splits.
- The student learns a teacher's labels. Its ceiling on the human-labelled dev set is
  therefore bounded by teacher quality, and the dev set is what makes the number meaningful.
- Training clips that also appear in the dev benchmark are removed (309 clip ids). Validation
  for checkpoint selection is a clip-level 10% holdout of the pseudo-labelled data, so the
  human-labelled dev set is never used for training or model selection.
- 12 classes: car, truck, bus, autorickshaw, motorcycle, bicycle, rider, person, animal,
  traffic light, traffic sign, vehicle fallback. Prepared training set: 3741 train / 259 val
  frames, 17,442 boxes (stored at max side 960 px).
- Segmentation / mIoU from the plan is **not done**. Masks in the training data are also model
  generated and there is no human-labelled segmentation split here.

## Metrics

- 12-class COCO-style mAP@0.5:0.95 and mAP@0.5 on the dev split, ignore regions respected
  (`idd/metrics.py`, adapted from the benchmark's own `evaluate.py`).
- 8-class "leaderboard score" from the unmodified upstream script, so it can be set against
  the published baselines in the benchmark card (YOLO11n 19.7, YOLOv8n 19.3, YOLO26x 34.2).
  Only 8 COCO-nameable classes are scored there; rider is merged into person.
- Classes without any dev ground truth are excluded from means. Small classes are noisy:
  animal has 10 GT boxes, traffic light 20, bicycle 22, vehicle fallback 12.

## Results

### COCO-pretrained YOLO11n, no fine-tuning (done, GPU, 640 px, 450 dev frames)

| Metric | Value |
|---|---|
| 8-class leaderboard score (mAP50-95) | 18.9 (upstream card reports 19.7 for the same model; my pipeline differs slightly in letterboxing and thresholds) |
| 8-class AP50 / near | 32.6 / 19.8 |
| 12-class mAP50-95 / mAP50 | 11.9 / 20.3 (COCO cannot name autorickshaw, rider, traffic sign, vehicle fallback, so these score 0 by construction) |
| Auto-rickshaws found by any vehicle box | 51% of 153 (called truck 27%, bus 12%, car 12%) |
| 12-class mAP by condition | day 14.3, dusk 9.0, night 12.8, rain 16.1 |

Per-class AP is in `results/coco_yolo11n_baseline.json`. Autorickshaw AP is 0.0 and car is 48.9.

### Fine-tuned YOLO11n (pending, run later)

| Metric | Value |
|---|---|
| 12-class mAP50-95 / mAP50 | pending |
| 8-class leaderboard score | pending |
| Autorickshaw / animal / rider AP | pending |
| Unmarked road edge | not a class in this data; not measured |

### Accuracy and latency per precision (pending, run later)

Batch 1, 640x640, single stream. `model` is one `session.run`; `e2e` adds letterbox and decode+NMS.

| Model | Provider | mAP50-95 | mAP50 | model p50 / p95 ms | FPS | e2e p95 ms | Size MB |
|---|---|---|---|---|---|---|---|
| FP32 | CPU | pending | pending | pending | pending | pending | pending |
| FP32 | CUDA | pending | pending | pending | pending | pending | pending |
| FP16 | CUDA | pending | pending | pending | pending | pending | pending |
| INT8 QDQ | CPU | pending | pending | pending | pending | pending | pending |
| INT8 QDQ, head FP32 | CPU | pending | pending | pending | pending | pending | pending |

Hardware: RTX 2050 4 GB laptop GPU, 12-thread CPU, onnxruntime-gpu 1.29. Expect INT8 QDQ on
CUDA EP to be no faster than FP16 (it falls back to fake-quant); INT8 is benchmarked on CPU.

## Usage (after `run_all.sh` has produced exports)

```
python -m idd.demo --model exports/yolo11n_indian_best_fp32.onnx --source 0                  # webcam
python -m idd.demo --model exports/int8_minmax.onnx --source clip.mp4 --out demo.mp4 --no-show
python -m idd.report --out results/tables.md          # paste into the Results section
python -m idd.trt_build --onnx exports/yolo11n_indian_best_fp32.onnx --out exports/m_fp16.engine --fp16
```

## Not done / out of scope here

- TensorRT engines (FP16/INT8): script and notes exist (`idd/trt_build.py`, `docs/TENSORRT_JETSON.md`),
  but TensorRT is not set up on this machine, so nothing was built or measured.
- Jetson: no hardware. No edge-device number exists in this repo.
- NOIR: adapter and doc exist (`idd/noir_adapter.py`, `docs/NOIR.md`), untested, no demo video.
- Segmentation model and mIoU.
- Training uses 4 of 646 shards (about 4000 frames, about 0.6% of the data) and a nano model.
  On a shared 4 GB GPU the batch size dropped to 8.

## Layout

```
idd/classes.py    class list, COCO->ours map
idd/fetch.py      download benchmark + shards (no login)
idd/data.py       shards -> YOLO dataset, clip-level split, benchmark clip exclusion
idd/train.py      Ultralytics fine-tune
idd/export.py     ONNX FP32 / FP16
idd/quantize.py   ORT static INT8 (QDQ), real-image calibration, optional FP32 head
idd/infer.py      letterbox / ORT session / decode / NMS (no torch needed)
idd/metrics.py    ignore-region COCO AP
idd/evaluate.py   score any .pt / .onnx on the dev split
idd/benchmark.py  latency, p50/p95/p99, FPS, CPU and CUDA
idd/demo.py       video / webcam demo, boxes + FPS overlay
idd/trt_build.py  TensorRT engine build (only if tensorrt imports)
idd/report.py     result JSONs -> README markdown tables
idd/noir_adapter.py  drop-in detector for NOIR's tracker
docs/             TENSORRT_JETSON.md, NOIR.md
tests/            unit tests (written, not yet run)
run_all.sh        end-to-end
```

## Environment notes

Ran in the shared `ml` venv. Installed `ultralytics` (no deps), `onnx`, `onnxruntime-gpu`
(over the CPU wheel; it is a superset) and `ml_dtypes`.
Data and weights are not committed (`.gitignore`). Credit: ThirdEye Labs for both datasets
(CC BY 4.0) and for the evaluation protocol that `idd/metrics.py` follows.
