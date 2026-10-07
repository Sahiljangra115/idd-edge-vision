#!/usr/bin/env bash
# Full pipeline. Run from anywhere with the ml venv active.
#   GPU_LOCK=/path/gpu.lock EPOCHS=40 ./run_all.sh     real run (40 epochs, 640 px, batch 8, in 20-epoch chunks)
#   SMOKE=1 ./run_all.sh                               every stage on tiny settings (1 epoch, 20 frames)
# Each GPU stage takes $GPU_LOCK on its own so a shared GPU can interleave with other jobs.
# CPU stages (FP32 export, INT8 calibration, CPU eval and benchmark) run without the lock, 4 threads.
set -euo pipefail
cd "$(dirname "$0")"

GPU_LOCK=${GPU_LOCK:-${TMPDIR:-/tmp}/gpu.lock}
gpu() { flock "$GPU_LOCK" "$@"; }
export OMP_NUM_THREADS=4
T=4

if [ "${SMOKE:-0}" = 1 ]; then
  NAME=smoke; EX=exports_smoke; P=smoke_; LIMIT="--limit 20"; RUNS=20; QN=8
  TRAIN="--epochs 1 --fraction 0.05"
else
  NAME=yolo11n_indian; EX=exports; P=""; LIMIT=""; RUNS=200; QN=128
  TRAIN="--epochs ${EPOCHS:-40} --chunk ${CHUNK:-20} --patience 0"
fi

# data (skipped when already prepared)
if [ ! -f data/yolo/prep_summary.json ]; then
  python -m idd.fetch --shards 10 150 300 450
  python -m idd.data --shards data/shards/train-00010.tar data/shards/train-00150.tar \
      data/shards/train-00300.tar data/shards/train-00450.tar --out data/yolo
fi

# baseline: COCO-pretrained, no fine-tuning
[ -f results/coco_yolo11n_baseline.json ] || \
  gpu python -m idd.evaluate --model weights/yolo11n.pt --coco --name coco_yolo11n_baseline --provider cuda

# fine-tune. With --chunk each call trains at most CHUNK epochs, exits 3, and the next call resumes,
# so no single GPU-lock hold runs for the whole training.
while :; do
  rc=0; gpu python -m idd.train --name $NAME --batch 8 --workers 3 $TRAIN || rc=$?
  [ $rc -eq 0 ] && break
  [ $rc -eq 3 ] || exit $rc
done
M=weights/${NAME}_best
gpu python -m idd.evaluate --model $M.pt --name ${P}pt_fp32 --provider cuda $LIMIT

# export: FP32 on CPU, FP16 needs the GPU
python -m idd.export --weights $M.pt --out $EX --precisions fp32
gpu python -m idd.export --weights $M.pt --out $EX --precisions fp16
F32=$EX/${NAME}_best_fp32.onnx
F16=$EX/${NAME}_best_fp16.onnx

# INT8 QDQ, real-frame calibration on CPU
python -m idd.quantize --model $F32 --n $QN --out $EX/int8_minmax.onnx
python -m idd.quantize --model $F32 --n $QN --out $EX/int8_minmax_headfp32.onnx --keep-head-fp32
python -m idd.quantize --model $F32 --n $QN --out $EX/int8_pct_headfp32.onnx --method percentile --keep-head-fp32

# accuracy per precision on the human-labelled dev split
python -m idd.evaluate --model $F32 --name ${P}onnx_fp32_cpu --provider cpu --threads $T $LIMIT
for q in int8_minmax int8_minmax_headfp32 int8_pct_headfp32; do
  python -m idd.evaluate --model $EX/$q.onnx --name ${P}onnx_${q}_cpu --provider cpu --threads $T $LIMIT
done
gpu python -m idd.evaluate --model $F32 --name ${P}onnx_fp32_cuda --provider cuda $LIMIT
gpu python -m idd.evaluate --model $F16 --name ${P}onnx_fp16_cuda --provider cuda $LIMIT

# TensorRT through the ONNX Runtime TensorRT EP, only when TRT 10 libs are in ./.pydeps/tensorrt_libs
TRT=0; ls .pydeps/tensorrt_libs/libnvinfer.so.10* >/dev/null 2>&1 && TRT=1
if [ $TRT = 1 ]; then
  gpu python -m idd.evaluate --model $F32 --name ${P}trt_fp32 --provider trt_fp32 $LIMIT
  gpu python -m idd.evaluate --model $F32 --name ${P}trt_fp16 --provider trt_fp16 $LIMIT
  gpu python -m idd.evaluate --model $EX/int8_minmax.onnx --name ${P}trt_int8_minmax --provider trt_int8 $LIMIT
fi

# latency: CPU without the lock (4 threads), CUDA under it
python -m idd.benchmark --models $F32 $EX/int8_*.onnx --providers cpu --threads $T --runs $RUNS \
    --out results/${P}latency_cpu.json
gpu python -m idd.benchmark --models $F32 $F16 $EX/int8_minmax_headfp32.onnx --providers cuda --runs $RUNS \
    --out results/${P}latency_cuda.json
[ $TRT = 1 ] && gpu python -m idd.benchmark --models $F32 --providers trt_fp32 trt_fp16 --runs $RUNS \
    --out results/${P}latency_trt.json
[ $TRT = 1 ] && gpu python -m idd.benchmark --models $EX/int8_minmax.onnx --providers trt_int8 --runs $RUNS \
    --out results/${P}latency_trt_int8.json

# per-class error analysis and README tables
python -m idd.analyze --runs coco_yolo11n_baseline ${P}pt_fp32 --out results/${P}error_analysis.md
python -m idd.report --prefix "$P" --out results/${P}tables.md
