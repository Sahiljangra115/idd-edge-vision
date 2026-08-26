# TensorRT and Jetson notes

Status: nothing here has been run. No TensorRT install on the dev laptop, no Jetson hardware.

## Rules that matter

- Build the engine on the device that will run it. Engines are tied to GPU architecture and TensorRT version.
- Start from the FP32 ONNX (`exports/*_fp32.onnx`), not the ORT-quantized QDQ file. TensorRT does its own
  INT8 calibration; feeding it ORT's QDQ model is a different (explicit quantization) path.
- Use the same pre/post-processing as `idd/infer.py`. Engine output is the same raw head, shape (1, 16, 8400).

## Build

Script (only runs if `tensorrt` imports, otherwise prints this guidance):

    python -m idd.trt_build --onnx exports/yolo11n_indian_best_fp32.onnx --out exports/model_fp16.engine --fp16
    python -m idd.trt_build --onnx exports/yolo11n_indian_best_fp32.onnx --out exports/model_int8.engine --int8 --calib data/bench/images/dev

INT8 needs `pycuda`. Calibrate on real frames; 200 is plenty for a nano model. Keep calibration frames
disjoint from the eval frames if you want an honest accuracy number (the dev frames are the default
only for convenience).

## Jetson (JetPack)

TensorRT, CUDA and cuDNN come with JetPack. `trtexec` is the quickest route:

    /usr/src/tensorrt/bin/trtexec --onnx=yolo11n_indian_best_fp32.onnx --saveEngine=model_fp16.engine --fp16
    /usr/src/tensorrt/bin/trtexec --loadEngine=model_fp16.engine --warmUp=500 --iterations=200 --percentile=95

`trtexec` prints GPU compute latency percentiles, which is not the same as the e2e number in
`idd/benchmark.py` (no letterbox, no NMS). Report both and label them.
Set power mode first (`sudo nvpmodel -m 0 && sudo jetson_clocks`) and write the mode in the results.

## Measuring honestly

- Report p50 and p95, batch 1, after warm-up, with the power mode and clocks stated.
- Run mAP on the engine with the same dev split before claiming an INT8 accuracy number.
- Alternative on Jetson: onnxruntime-gpu with the TensorRT execution provider, which avoids a separate engine step.
