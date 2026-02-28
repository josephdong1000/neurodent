## Testing

Always run the test suite before finalizing changes:

```bash
uv run pytest tests/ --cov=neurodent --cov-report=term-missing -v
```

All tests must pass and new code should include appropriate test coverage.