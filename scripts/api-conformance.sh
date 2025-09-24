#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Simple test runner for API conformance tests
# Runs pytest with snapshot testing for Pydantic models

# Get the script directory
THIS_DIR=$(dirname "$0")
ROOT_DIR="$THIS_DIR/.."

cd "$ROOT_DIR"

# Run pytest with snapshot testing
echo "=== Running API Conformance Tests ==="
pytest -s -v tests/ \
    --snapshot-update \
    "$@"

echo "✅ API Conformance Tests Complete"
