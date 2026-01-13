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

PACKAGE_DIR="${1:-src/llama_stack}"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "ERROR: Package directory '$PACKAGE_DIR' does not exist"
    exit 1
fi

# Get all directories with Python files (excluding __init__.py)
# Use a portable approach that works with sh, bash, and zsh
temp_file=$(mktemp)
trap 'rm -f "$temp_file"' EXIT

find "$PACKAGE_DIR" \
    -type f \
    -name "*.py" ! -name "__init__.py" \
    ! -path "*/.venv/*" \
    ! -path "*/node_modules/*" \
    -exec dirname {} + | sort -u | while IFS= read -r dir; do
    if [ ! -f "$dir/__init__.py" ]; then
        echo "ERROR: Missing __init__.py in directory: $dir"
        echo "This directory contains Python files but no __init__.py, which may cause packaging issues."
        echo "1" > "$temp_file"
    fi
done

if [ -f "$temp_file" ] && [ -s "$temp_file" ]; then
    exit 1
else
    exit 0
fi
