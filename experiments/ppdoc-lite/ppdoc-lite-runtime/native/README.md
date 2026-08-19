# Native Paddle quality-tier experiment

This directory is an isolated fallback for the exact RT-DETR teacher. It is
not the fast/default PPdoc runtime. S and M stay on the direct Rust/ONNX Runtime
path. The teacher needs this fallback because both Paddle2ONNX and OpenVINO
changed its decoded coordinates and failed the raw-tensor fidelity gate.

## Pinned Windows runtime

- Paddle Inference: 3.2.0, CPU/AVX/MKL, MSVC 2019
- upstream commit: `e22e2f9af7eeced7e3c9582ddb69a617887d3eb9`
- archive: <https://paddle-inference-lib.bj.bcebos.com/3.2.0/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2019/paddle_inference.zip>
- archive SHA-256: `23a2ea41abaedb7dfb928dc10baa72975d50b7a8ffe28f8e081a16a8977a95b2`

The archive must match the 3.2.0 Paddle version that exported the teacher's PIR
`inference.json`. Paddle Inference 3.0.0 cannot parse that graph.

The runtime uses the official C++ API pattern: one persistent predictor,
optional oneDNN, explicit CPU thread count, and a one-shape oneDNN cache. It deliberately
does not call `EnableMemoryOptim()`: for this Paddle 3.2 PIR graph that API asks
for the absent legacy `memory_optimize_pass` and aborts predictor creation.
`EnableONEDNN()` is used instead of its deprecated `EnableMKLDNN()` alias.

Official references:

- <https://www.paddlepaddle.org.cn/inference/master/guides/quick_start/cpp_demo.html>
- <https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/Config/CPUConfig.html>
- <https://www.paddlepaddle.org.cn/inference/master/api_reference/cxx_api_doc/Config/OtherFunction.html>

## Dependency boundary

`prepare_windows_runtime.ps1` copies only the executable's required non-system
DLLs. Paddle loads `mklml.dll` lazily, so it is required even though it is absent
from the ordinary PE import table. The six upstream DLLs total 278,464,256
bytes; the source-built adapter is 142,336 bytes in the audited build. Model
files are separate hash-checked assets. No Paddle Python wheel, NumPy, PaddleX,
training code, compiler, Docker image, calibration data, or SDK headers ship.

`ppdoc_paddle.cpp` is the product C ABI adapter loaded by the Rust provider.
`paddle_probe.cpp` remains a measurement-only executable that consumes an
already prepared 800x800 CHW tensor so image decoding and resize can be measured
independently.

## Reproduce

```powershell
.\prepare_windows_runtime.ps1 `
  -SdkDir C:\sdk\paddle_inference `
  -OutputDir C:\scratch\ppdoc-paddle-runtime

cargo build --release --locked --no-default-features --features ppdoc
.\target\release\legalpdf.exe ppdoc-images C:\pages\page-1.png `
  --model-pack C:\models\teacher-boxonly `
  --runtime C:\scratch\ppdoc-paddle-runtime\ppdoc_paddle.dll `
  --threads 2
```

Generated SDKs, binaries, models, tensors, and benchmark outputs stay outside
Git. Only source, pinned manifests, and concise receipts are retained here.
