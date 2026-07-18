.PHONY: setup test docs docs-live docs-linkcheck clean requirements

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

# Check documentation links
docs-linkcheck:
	cd docs && uv run sphinx-build -b linkcheck . _build/linkcheck
	@echo "✅ Link check complete! See docs/_build/linkcheck/output.txt for results"

# Clean build artifacts
clean:
	rm -rf docs/_build
	rm -rf .pytest_cache
	rm -rf **/__pycache__

# Regenerate hash-pinned requirements from uv.lock (pyproject.toml is the source of truth)
requirements:
	uv export --frozen --format requirements-txt --no-emit-local --extra all -o requirements.txt >/dev/null
	uv export --frozen --format requirements-txt --no-emit-local --all-extras -o requirements-dev.txt >/dev/null
