#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Generate distribution configs in an isolated environment.
#
# Usage:
#   ./scripts/distro_generate_config.sh <distro_dir> [--overlay <overlay>] [--output <output>]
#
# Examples:
#   ./scripts/distro_generate_config.sh packages/llama-stack-distribution-demo \
#       --overlay packages/llama-stack-distribution-demo/patches/config.yaml \
#       --output packages/llama-stack-distribution-demo/src/llama_stack_distribution_demo/dist/config.yaml

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <distro_dir> [--overlay <overlay>] [--output <output>]" >&2
    exit 1
fi

DISTRO_DIR="$1"
shift

exec uv run --project "$DISTRO_DIR" llama stack config generate "$@"
