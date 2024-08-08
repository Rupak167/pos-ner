install:
	poetry install

run:
	docker compose up --build

clear:
	docker compose down --remove-orphans

train:
	python training.py

test:
	python test.py

prepare-onnx:
	python prepare_onnx_model.py

infer-onnx:
	python inference_onnx_model.py