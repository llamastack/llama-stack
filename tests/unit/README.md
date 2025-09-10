# Llama Stack Unit Tests

## Unit Tests

Unit tests verify individual components and functions in isolation. They are fast, reliable, and don't require external services.

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.12+ installed
2. **uv Package Manager**: Install `uv` if not already installed

You can run the unit tests by running:

```bash
./scripts/unit-tests.sh [PYTEST_ARGS]
```

Any additional arguments are passed to pytest. For example, you can specify a test directory, a specific test file, or any pytest flags (e.g., -vvv for verbosity). If no test directory is specified, it defaults to "tests/unit", e.g:

```bash
./scripts/unit-tests.sh tests/unit/registry/test_registry.py -vvv
```

If you'd like to run for a non-default version of Python (currently 3.12), pass `PYTHON_VERSION` variable as follows:

```
source .venv/bin/activate
PYTHON_VERSION=3.13 ./scripts/unit-tests.sh
```

### Test Configuration

- **Test Discovery**: Tests are automatically discovered in the `tests/unit/` directory
- **Async Support**: Tests use `--asyncio-mode=auto` for automatic async test handling
- **Coverage**: Tests generate coverage reports in `htmlcov/` directory
- **Python Version**: Defaults to Python 3.12, but can be overridden with `PYTHON_VERSION` environment variable

### Coverage Reports

After running tests, you can view coverage reports:

```bash
# Open HTML coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```
