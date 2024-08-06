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