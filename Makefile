.PHONY: train infer export-onnx bench test fmt

train:
	python -m src.train --config configs/default.yaml

infer:
	python -m src.infer --config configs/inference.yaml

export-onnx:
	python -m src.export_onnx --config configs/default.yaml

bench:
	python -m src.benchmark --config configs/inference.yaml

test:
	pytest

fmt:
	pre-commit run --all-files || true

