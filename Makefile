.PHONY: up down logs test fmt lint

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f api

test:
	# --build is not optional: the test image bakes in tests/ and app/, so
	# without it an edited file runs green against the previous build.
	docker compose --profile test run --rm --build tests

fmt:
	uv run ruff format .

lint:
	uv run ruff check .
