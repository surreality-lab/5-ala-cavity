.PHONY: install install-dev lint fix clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

fix:
	ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist
