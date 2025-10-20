#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

PYTHONPATH=${PYTHONPATH:-}
THIS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

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

stack_dir=$(dirname $(dirname $THIS_DIR))
PYTHONPATH=$PYTHONPATH:$stack_dir \
  python -m docs.openapi_generator.generate $(dirname $THIS_DIR)/static

cp $stack_dir/docs/static/stainless-llama-stack-spec.yaml $stack_dir/client-sdks/stainless/openapi.yml
