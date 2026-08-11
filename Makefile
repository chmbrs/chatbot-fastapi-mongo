.PHONY: up down logs test fmt lint

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	docker compose --profile test run --rm tests

fmt:
	uv run ruff format .

lint:
	uv run ruff check .
