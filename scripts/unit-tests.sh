#!/bin/sh

# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHON_VERSION=${PYTHON_VERSION:-3.12}

set -e

# Always generate the HTML report, even if tests fail
cleanup() {
    echo "Generating HTML coverage report..."
    uv run --python "$PYTHON_VERSION" coverage html -d htmlcov-$PYTHON_VERSION
}
trap cleanup EXIT

command -v uv >/dev/null 2>&1 || { echo >&2 "uv is required but it's not installed. Exiting."; exit 1; }

uv python find "$PYTHON_VERSION"
FOUND_PYTHON=$?
if [ $FOUND_PYTHON -ne 0 ]; then
     uv python install "$PYTHON_VERSION"
fi

# Run unit tests with coverage (pytest-cov is configured via pyproject.toml addopts).
# The --cov-fail-under flag enforces the minimum coverage threshold.
uv run --python "$PYTHON_VERSION" --with-editable . --group unit \
    pytest -s -v --cov-report=html:htmlcov-$PYTHON_VERSION tests/unit/ "$@"

# Generate text report for CI logs (fail_under is set in .coveragerc)
uv run --python "$PYTHON_VERSION" coverage report
