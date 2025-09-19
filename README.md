# Production ML Refactor Demo (GPU-ready)

**What this shows:** clean structure, Dockerized GPU runtime, typed code, config via YAML, logging, deterministic runs, ONNX export, and a simple benchmarking harness.

### Quickstart

```bash
# local (CPU)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make train
make infer
make export-onnx
make bench
```

### Docker (GPU)

```bash
# Requires NVIDIA Container Toolkit
docker build -t ml-refactor-demo:latest .
# Train on GPU (falls back to CPU if none)
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=0 ml-refactor-demo:latest make train
```

### Commands

* `make train` → trains a tiny CNN on synthetic MNIST-like data
* `make infer` → runs batched inference from `configs/inference.yaml`
* `make export-onnx` → exports model to `artifacts/model.onnx`
* `make bench` → latency/throughput benchmark (CPU/GPU)

Optional:

* Set `MLFLOW_TRACKING_URI` to enable auto-logging of metrics/artifacts.

