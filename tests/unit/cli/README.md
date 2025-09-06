# CLI Unit Tests

This directory contains unit tests for the Llama Stack CLI commands.

## Test Files

### `test_version.py`
Comprehensive unit tests for the `llama version` command using pytest and mocking. These tests cover:

- Package version retrieval
- Build information handling
- Component information retrieval
- JSON and table output formats
- Error handling scenarios
- Component type detection

**Requirements:** pytest, unittest.mock

**Run with:** `pytest tests/unit/cli/test_version.py -v`

### `test_version_simple.py`
Simple unit tests that can run without external dependencies. These tests verify:

- File structure and existence
- Code structure and imports
- Logic validation
- Configuration correctness

**Requirements:** None (uses only standard library)

**Run with:** `python tests/unit/cli/test_version_simple.py`

### `test_version_integration.py`
Integration tests that test the actual CLI command execution. These tests:

- Execute the actual CLI commands
- Verify command-line interface
- Test real output formats
- Validate build script execution

**Requirements:** Full Llama Stack environment

**Run with:** `python tests/unit/cli/test_version_integration.py`

### `test_stack_config.py`
Tests for stack configuration parsing and upgrading.

## Running Tests

### Run All CLI Tests
```bash
# With pytest (if available)
pytest tests/unit/cli/ -v

# Simple tests only (no dependencies)
python tests/unit/cli/test_version_simple.py
```

### Run Specific Tests
```bash
# Run version command tests
python tests/unit/cli/test_version_simple.py

# Run integration tests (requires full environment)
python tests/unit/cli/test_version_integration.py
```

## Test Coverage

The version command tests cover:

- ✅ Package version detection
- ✅ Build information retrieval
- ✅ Component discovery and listing
- ✅ JSON output format
- ✅ Table output format
- ✅ Error handling
- ✅ CLI integration
- ✅ Build script functionality
- ✅ File structure validation

## Adding New Tests

When adding new CLI commands or functionality:

1. Create unit tests following the pattern in `test_version.py`
2. Add simple validation tests in a `test_<command>_simple.py` file
3. Consider integration tests for end-to-end validation
4. Update this README with the new test information

## Notes

- Simple tests are preferred for CI/CD as they have no external dependencies
- Integration tests are useful for local development and full validation
- Mock-based tests provide comprehensive coverage of edge cases
