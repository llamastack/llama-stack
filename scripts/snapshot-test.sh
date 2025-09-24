#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -eu

# Simple test runner for snapshot tests
# Runs pytest with snapshot testing to detect schema changes

# Get the script directory
THIS_DIR=$(dirname "$0")
ROOT_DIR="$THIS_DIR/.."
SCRIPT=""
REGENERATE_SNAPSHOT=""

cd "$ROOT_DIR"

usage() {
    cat << EOF
Usage: $0 [OPTIONS] <test-script> [regenerate-snapshot]

Arguments:
    test-script              Path to the test script to run (required)
    regenerate-snapshot      Set to "true" to regenerate snapshots (optional)

Options:
    --help                   Show this help message

Examples:
    # Run snapshot test
    $0 tests/api/test_pydantic_models.py

    # Run with snapshot regeneration
    $0 tests/api/test_pydantic_models.py true
EOF
    exit 0
}

# Parse command line arguments
if [[ $# -gt 0 ]] && [[ "$1" == "--help" ]]; then
    usage
fi

if [[ $# -lt 1 ]]; then
    echo "Error: Missing required test script argument"
    usage
fi

SCRIPT="$1"
REGENERATE_SNAPSHOT="${2:-false}"

# Run pytest with snapshot testing
echo "=== Running Snapshot Tests ==="
  if [[ "$REGENERATE_SNAPSHOT" == "true" ]]; then
    echo "Regenerating snapshots..."
    pytest -s -v "$SCRIPT" --snapshot-update
    exit 0
  else
    pytest -s -v "$SCRIPT"  # do not update snapshots.
  fi

echo "✅ Snapshot Tests Complete"
