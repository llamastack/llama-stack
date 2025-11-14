#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# Pass --target-root to direct generated artifacts into an alternate checkout
# (used by the trusted autofix workflow running from a base-branch worktree).

PYTHONPATH=${PYTHONPATH:-}
THIS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SOURCE_STACK_DIR="$(dirname "$(dirname "$THIS_DIR")")"
TARGET_STACK_DIR="$SOURCE_STACK_DIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-root)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Error: --target-root requires a value" >&2
                exit 1
            fi
            TARGET_STACK_DIR="$1"
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            exit 1
            ;;
    esac
    shift
done
TARGET_STATIC_DIR="$TARGET_STACK_DIR/docs/openapi_generator/static"
TARGET_SPEC_PATH="$TARGET_STACK_DIR/client-sdks/stainless/openapi.yml"

set -euo pipefail

missing_packages=()

check_package() {
    if ! pip show "$1" &>/dev/null; then
        missing_packages+=("$1")
    fi
}

if [ ${#missing_packages[@]} -ne 0 ]; then
    echo "Error: The following package(s) are not installed:"
    printf " - %s\n" "${missing_packages[@]}"
    echo "Please install them using:"
    echo "pip install ${missing_packages[*]}"
    exit 1
fi

mkdir -p "$TARGET_STATIC_DIR"
mkdir -p "$(dirname "$TARGET_SPEC_PATH")"

PYTHONPATH="$TARGET_STACK_DIR:$TARGET_STACK_DIR/src:$PYTHONPATH" \
  python "$SOURCE_STACK_DIR/docs/openapi_generator/generate.py" "$TARGET_STATIC_DIR"

cp "$TARGET_STACK_DIR/docs/static/stainless-llama-stack-spec.yaml" "$TARGET_SPEC_PATH"
