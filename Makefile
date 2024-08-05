install:
	poetry install

run:
	docker compose up --build

clear:
	docker compose down --remove-orphans