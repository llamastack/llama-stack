#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

echo "Running comprehensive GuideLL benchmark suite..."
echo "Start time: $(date)"

# Common parameters
MAX_SECONDS=60
PROMPT_TOKENS=512
OUTPUT_TOKENS=256

# Define benchmark configurations: (target, stack_replicas, vllm_replicas)
configs=(
    # "stack 1 1"
    # "stack 2 1"
    # "stack 4 1"
    # "vllm 1 1"
    # "stack 2 2"
    # "stack 4 2"
    "stack 8 2"
    # "vllm 1 2"
)

for config in "${configs[@]}"; do
    read -r target stack_replicas vllm_replicas <<< "$config"

    echo ""
    echo "=========================================="
    echo "Running benchmark: $target (stack=$stack_replicas, vllm=$vllm_replicas)"
    echo "Start: $(date)"
    echo "=========================================="

    if [[ "$target" == "vllm" ]]; then
        # For vLLM target, only use vllm-replicas parameter
        ./run-guidellm-benchmark.sh \
            --target "$target" \
            --vllm-replicas "$vllm_replicas" \
            --max-seconds "$MAX_SECONDS" \
            --prompt-tokens "$PROMPT_TOKENS" \
            --output-tokens "$OUTPUT_TOKENS"
    else
        # For stack target, use both parameters
        ./run-guidellm-benchmark.sh \
            --target "$target" \
            --stack-replicas "$stack_replicas" \
            --vllm-replicas "$vllm_replicas" \
            --max-seconds "$MAX_SECONDS" \
            --prompt-tokens "$PROMPT_TOKENS" \
            --output-tokens "$OUTPUT_TOKENS"
    fi

    echo "Completed: $(date)"
    echo "Waiting 30 seconds before next benchmark..."
    sleep 30
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results files generated:"
ls -la guidellm-*.txt guidellm-*.json 2>/dev/null || echo "No result files found"
