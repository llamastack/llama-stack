#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Integration test runner script for Llama Stack
# This script extracts the integration test logic from GitHub Actions
# to allow developers to run integration tests locally

# Default values
STACK_CONFIG=""
TEST_SUITE="base"
TEST_SETUP=""
TEST_SUBDIRS=""
TEST_PATTERN=""
INFERENCE_MODE="replay"
EXTRA_PARAMS=""

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --stack-config STRING    Stack configuration to use (required)
    --suite STRING           Test suite to run (default: 'base')
    --setup STRING           Test setup (models, env) to use (e.g., 'ollama', 'ollama-vision', 'gpt', 'vllm')
    --inference-mode STRING  Inference mode: record or replay (default: replay)
    --subdirs STRING         Comma-separated list of test subdirectories to run (overrides suite)
    --pattern STRING         Regex pattern to pass to pytest -k
    --help                   Show this help message

Suites are defined in tests/integration/suites.py and define which tests to run.
Setups are defined in tests/integration/setups.py and provide global configuration (models, env).

You can also specify subdirectories (of tests/integration) to select tests from, which will override the suite.

Examples:
    # Basic inference tests with ollama
    $0 --stack-config server:ci-tests --suite base --setup ollama

    # Multiple test directories with vllm
    $0 --stack-config server:ci-tests --subdirs 'inference,agents' --setup vllm

    # Vision tests with ollama
    $0 --stack-config server:ci-tests --suite vision  # default setup for this suite is ollama-vision

    # Record mode for updating test recordings
    $0 --stack-config server:ci-tests --suite base --inference-mode record
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-config)
            STACK_CONFIG="$2"
            shift 2
            ;;
        --setup)
            TEST_SETUP="$2"
            shift 2
            ;;
        --subdirs)
            TEST_SUBDIRS="$2"
            shift 2
            ;;
        --suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        --inference-mode)
            INFERENCE_MODE="$2"
            shift 2
            ;;
        --pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done


# Validate required parameters
if [[ -z "$STACK_CONFIG" ]]; then
    echo "Error: --stack-config is required"
    usage
    exit 1
fi

if [[ -z "$TEST_SETUP" && -n "$TEST_SUBDIRS" ]]; then
    echo "Error: --test-setup is required when --test-subdirs is provided"
    usage
    exit 1
fi

if [[ -z "$TEST_SUITE" && -z "$TEST_SUBDIRS" ]]; then
    echo "Error: --test-suite or --test-subdirs is required"
    exit 1
fi

echo "=== Llama Stack Integration Test Runner ==="
echo "Stack Config: $STACK_CONFIG"
echo "Setup: $TEST_SETUP"
echo "Inference Mode: $INFERENCE_MODE"
echo "Test Suite: $TEST_SUITE"
echo "Test Subdirs: $TEST_SUBDIRS"
echo "Test Pattern: $TEST_PATTERN"
echo ""

echo "Checking llama packages"
uv pip list | grep llama

# Check storage and memory before tests
echo "=== System Resources Before Tests ==="
free -h 2>/dev/null || echo "free command not available"
df -h
echo ""

# Set environment variables
export LLAMA_STACK_CLIENT_TIMEOUT=300

THIS_DIR=$(dirname "$0")

if [[ -n "$TEST_SETUP" ]]; then
    EXTRA_PARAMS="--setup=$TEST_SETUP"
fi

# Apply setup-specific environment variables (needed for server startup and tests)
echo "=== Applying Setup Environment Variables ==="

# the server needs this
export LLAMA_STACK_TEST_INFERENCE_MODE="$INFERENCE_MODE"

SETUP_ENV=$(PYTHONPATH=$THIS_DIR/.. python "$THIS_DIR/get_setup_env.py" --suite "$TEST_SUITE" --setup "$TEST_SETUP" --format bash)
echo "Setting up environment variables:"
echo "$SETUP_ENV"
eval "$SETUP_ENV"
echo ""

ROOT_DIR="$THIS_DIR/.."
cd $ROOT_DIR

# check if "llama" and "pytest" are available. this script does not use `uv run` given
# it can be used in a pre-release environment where we have not been able to tell
# uv about pre-release dependencies properly (yet).
if ! command -v llama &> /dev/null; then
    echo "llama could not be found, ensure llama-stack is installed"
    exit 1
fi

if ! command -v pytest &> /dev/null; then
    echo "pytest could not be found, ensure pytest is installed"
    exit 1
fi

# Start Llama Stack Server if needed
if [[ "$STACK_CONFIG" == *"server:"* ]]; then
    stop_server() {
        echo "Stopping Llama Stack Server..."
        pids=$(lsof -i :8321 | awk 'NR>1 {print $2}')
        if [[ -n "$pids" ]]; then
            echo "Killing Llama Stack Server processes: $pids"
            kill -9 $pids
        else
            echo "No Llama Stack Server processes found ?!"
        fi
        echo "Llama Stack Server stopped"
    }

    # check if server is already running
    if curl -s http://localhost:8321/v1/health 2>/dev/null | grep -q "OK"; then
        echo "Llama Stack Server is already running, skipping start"
    else
        echo "=== Starting Llama Stack Server ==="
        nohup llama stack run ci-tests --image-type venv > server.log 2>&1 &

        echo "Waiting for Llama Stack Server to start..."
        for i in {1..30}; do
            if curl -s http://localhost:8321/v1/health 2>/dev/null | grep -q "OK"; then
                echo "✅ Llama Stack Server started successfully"
                break
            fi
            if [[ $i -eq 30 ]]; then
                echo "❌ Llama Stack Server failed to start"
                echo "Server logs:"
                cat server.log
                exit 1
            fi
            sleep 1
        done
        echo ""
    fi

    trap stop_server EXIT ERR INT TERM
fi

# Run tests
echo "=== Running Integration Tests ==="
EXCLUDE_TESTS="builtin_tool or safety_with_image or code_interpreter or test_rag"

# Additional exclusions for vllm setup
if [[ "$TEST_SETUP" == "vllm" ]]; then
    EXCLUDE_TESTS="${EXCLUDE_TESTS} or test_inference_store_tool_calls"
fi

PYTEST_PATTERN="not( $EXCLUDE_TESTS )"
if [[ -n "$TEST_PATTERN" ]]; then
    PYTEST_PATTERN="${PYTEST_PATTERN} and $TEST_PATTERN"
fi

echo "Test subdirs to run: $TEST_SUBDIRS"

if [[ -n "$TEST_SUBDIRS" ]]; then
    # Collect all test files for the specified test types
    TEST_FILES=""
    for test_subdir in $(echo "$TEST_SUBDIRS" | tr ',' '\n'); do
        if [[ -d "tests/integration/$test_subdir" ]]; then
            # Find all Python test files in this directory
            test_files=$(find tests/integration/$test_subdir -name "test_*.py" -o -name "*_test.py")
            if [[ -n "$test_files" ]]; then
                TEST_FILES="$TEST_FILES $test_files"
                echo "Added test files from $test_subdir: $(echo $test_files | wc -w) files"
            fi
        else
            echo "Warning: Directory tests/integration/$test_subdir does not exist"
        fi
    done

    if [[ -z "$TEST_FILES" ]]; then
        echo "No test files found for the specified test types"
        exit 1
    fi

    echo ""
    echo "=== Running all collected tests in a single pytest command ==="
    echo "Total test files: $(echo $TEST_FILES | wc -w)"

    PYTEST_TARGET="$TEST_FILES"
else
    PYTEST_TARGET="tests/integration/"
    EXTRA_PARAMS="$EXTRA_PARAMS --suite=$TEST_SUITE"
fi

set +e
set -x
pytest -s -v $PYTEST_TARGET \
    --stack-config="$STACK_CONFIG" \
    --inference-mode="$INFERENCE_MODE" \
    -k "$PYTEST_PATTERN" \
    $EXTRA_PARAMS \
    --color=yes \
    --capture=tee-sys
exit_code=$?
set +x
set -e

if [ $exit_code -eq 0 ]; then
    echo "✅ All tests completed successfully"
elif [ $exit_code -eq 5 ]; then
    echo "⚠️ No tests collected (pattern matched no tests)"
else
    echo "❌ Tests failed"
    exit 1
fi

# Check storage and memory after tests
echo ""
echo "=== System Resources After Tests ==="
free -h 2>/dev/null || echo "free command not available"
df -h

echo ""
echo "=== Integration Tests Complete ==="
