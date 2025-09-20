#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Script to profile an already running Llama Stack server
# Usage: ./profile_running_server.sh [duration_seconds] [output_file]

DURATION=${1:-60}  # Default 60 seconds
OUTPUT_FILE=${2:-"llama_stack_profile"}  # Default output file

echo "Looking for running Llama Stack server..."

# Find the server PID
SERVER_PID=$(ps aux | grep "llama_stack.core.server.server" | grep -v grep | awk '{print $2}' | head -1)


if [ -z "$SERVER_PID" ]; then
    echo "Error: No running Llama Stack server found"
    echo "Please start your server first with:"
    echo "LLAMA_STACK_LOGGING=\"all=ERROR\" MOCK_INFERENCE_URL=http://localhost:8080 SAFETY_MODEL=llama-guard3:1b uv run --with llama-stack python -m llama_stack.core.server.server docs/source/distributions/k8s-benchmark/stack_run_config.yaml"
    exit 1
fi

echo "Found Llama Stack server with PID: $SERVER_PID"

# Start py-spy profiling
echo "Starting py-spy profiling for ${DURATION} seconds..."
echo "Output will be saved to: ${OUTPUT_FILE}.svg"
echo ""
echo "You can now run your load test..."
echo ""

# Get the full path to py-spy
PYSPY_PATH=$(which py-spy)

# Check if running as root, if not, use sudo
if [ "$EUID" -ne 0 ]; then
    echo "py-spy requires root permissions on macOS. Running with sudo..."
    sudo "$PYSPY_PATH" record -o "${OUTPUT_FILE}.svg" -d ${DURATION} -p $SERVER_PID
else
    "$PYSPY_PATH" record -o "${OUTPUT_FILE}.svg" -d ${DURATION} -p $SERVER_PID
fi

echo ""
echo "Profiling completed! Results saved to: ${OUTPUT_FILE}.svg"
echo ""
echo "To view the flame graph:"
echo "open ${OUTPUT_FILE}.svg"
