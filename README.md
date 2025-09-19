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
