#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# Check for missing __init__.py files in Python packages
# This script finds directories that contain Python files but are missing __init__.py

set -euo pipefail

PACKAGE_DIR="${1:-llama_stack}"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "ERROR: Package directory '$PACKAGE_DIR' does not exist"
    exit 1
fi

missing_init_files=0

# Get all directories with Python files (excluding __init__.py) and check each one
while IFS= read -r -d '' dir; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "ERROR: Missing __init__.py in directory: $dir"
        echo "This directory contains Python files but no __init__.py, which may cause packaging issues."
        missing_init_files=1
    fi
done < <(
    find "$PACKAGE_DIR" \
        -type f \
        -name "*.py" ! -name "__init__.py" \
        ! -path "*/.venv/*" \
        ! -path "*/node_modules/*" \
        -exec dirname {} \; | sort -u | tr '\n' '\0'
)

exit $missing_init_files
