.PHONY: setup test docs docs-live clean

# Development setup - run this after cloning
setup:
	uv sync --all-extras
	uv run pre-commit install
	@echo "✅ Setup complete!"

# Run tests
test:
	uv run pytest

# Build documentation locally
docs:
	cd docs && uv run sphinx-build -b html . _build/html

# Serve docs with auto-reload
docs-live:
	cd docs && uv run sphinx-autobuild . _build/html

# Clean build artifacts
clean:
	rm -rf docs/_build
	rm -rf .pytest_cache
	rm -rf **/__pycache__
