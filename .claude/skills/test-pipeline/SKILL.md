---
name: test-pipeline
description: Run full test suite and lint checks.
---

Run the ActionFlow quality checks:

1. Run all tests: `PYTHONPATH=src python -m pytest tests/ -v`
2. Run linter: `ruff check src/ tests/`
3. If any tests fail, investigate and fix them
4. If lint issues are found, fix with `ruff check --fix src/ tests/` then verify
5. Report final pass/fail status
