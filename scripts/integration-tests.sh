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
PROVIDER=""
TEST_SUBDIRS=""
TEST_PATTERN=""
RUN_VISION_TESTS="false"
INFERENCE_MODE="replay"
EXTRA_PARAMS=""

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --stack-config STRING    Stack configuration to use (required)
    --provider STRING        Provider to use (ollama, vllm, etc.) (required)
    --test-subdirs STRING    Comma-separated list of test subdirectories to run (default: 'inference')
    --run-vision-tests       Run vision tests instead of regular tests
    --inference-mode STRING  Inference mode: record or replay (default: replay)
    --test-pattern STRING    Regex pattern to pass to pytest -k
    --help                   Show this help message

Examples:
    # Basic inference tests with ollama
    $0 --stack-config server:ci-tests --provider ollama

    # Multiple test directories with vllm
    $0 --stack-config server:ci-tests --provider vllm --test-subdirs 'inference,agents'

    # Vision tests with ollama
    $0 --stack-config server:ci-tests --provider ollama --run-vision-tests

    # Record mode for updating test recordings
    $0 --stack-config server:ci-tests --provider ollama --inference-mode record
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-config)
            STACK_CONFIG="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --test-subdirs)
            TEST_SUBDIRS="$2"
            shift 2
            ;;
        --run-vision-tests)
            RUN_VISION_TESTS="true"
            shift
            ;;
        --inference-mode)
            INFERENCE_MODE="$2"
            shift 2
            ;;
        --test-pattern)
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

if [[ -z "$PROVIDER" ]]; then
    echo "Error: --provider is required"
    usage
    exit 1
fi

echo "=== Llama Stack Integration Test Runner ==="
echo "Stack Config: $STACK_CONFIG"
echo "Provider: $PROVIDER"
echo "Test Subdirs: $TEST_SUBDIRS"
echo "Vision Tests: $RUN_VISION_TESTS"
echo "Inference Mode: $INFERENCE_MODE"
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
export LLAMA_STACK_TEST_INFERENCE_MODE="$INFERENCE_MODE"

# Configure provider-specific settings
if [[ "$PROVIDER" == "ollama" ]]; then
    export OLLAMA_URL="http://0.0.0.0:11434"
    export TEXT_MODEL="ollama/llama3.2:3b-instruct-fp16"
    export SAFETY_MODEL="ollama/llama-guard3:1b"
    EXTRA_PARAMS="--safety-shield=llama-guard"
else
    export VLLM_URL="http://localhost:8000/v1"
    export TEXT_MODEL="vllm/meta-llama/Llama-3.2-1B-Instruct"
    EXTRA_PARAMS=""
fi

THIS_DIR=$(dirname "$0")
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
fi

# Run tests
echo "=== Running Integration Tests ==="
EXCLUDE_TESTS="builtin_tool or safety_with_image or code_interpreter or test_rag"

# Additional exclusions for vllm provider
if [[ "$PROVIDER" == "vllm" ]]; then
    EXCLUDE_TESTS="${EXCLUDE_TESTS} or test_inference_store_tool_calls"
fi

PYTEST_PATTERN="not( $EXCLUDE_TESTS )"
if [[ -n "$TEST_PATTERN" ]]; then
    PYTEST_PATTERN="${PYTEST_PATTERN} and $TEST_PATTERN"
fi

# Run vision tests if specified
if [[ "$RUN_VISION_TESTS" == "true" ]]; then
    echo "Running vision tests..."
    set +e
    pytest -s -v tests/integration/inference/test_vision_inference.py \
        --stack-config="$STACK_CONFIG" \
        -k "$PYTEST_PATTERN" \
        --vision-model=ollama/llama3.2-vision:11b \
        --embedding-model=sentence-transformers/all-MiniLM-L6-v2 \
        --color=yes $EXTRA_PARAMS \
        --capture=tee-sys
    exit_code=$?
    set -e

    if [ $exit_code -eq 0 ]; then
        echo "✅ Vision tests completed successfully"
    elif [ $exit_code -eq 5 ]; then
        echo "⚠️ No vision tests collected (pattern matched no tests)"
    else
        echo "❌ Vision tests failed"
        exit 1
    fi
    exit 0
fi

# Run regular tests
if [[ -z "$TEST_SUBDIRS" ]]; then
   TEST_SUBDIRS=$(find tests/integration -maxdepth 1 -mindepth 1 -type d |
            sed 's|tests/integration/||' |
            grep -Ev "^(__pycache__|fixtures|test_cases|recordings|non_ci|post_training)$" |
            sort)
fi
echo "Test subdirs to run: $TEST_SUBDIRS"

# Collect all test files for the specified test types
TEST_FILES=""
for test_subdir in $(echo "$TEST_SUBDIRS" | tr ',' '\n'); do
    # Skip certain test types for vllm provider
    if [[ "$PROVIDER" == "vllm" ]]; then
        if [[ "$test_subdir" == "safety" ]] || [[ "$test_subdir" == "post_training" ]] || [[ "$test_subdir" == "tool_runtime" ]]; then
            echo "Skipping $test_subdir for vllm provider"
            continue
        fi
    fi

    if [[ "$STACK_CONFIG" != *"server:"* ]] && [[ "$test_subdir" == "batches" ]]; then
        echo "Skipping $test_subdir for library client until types are supported"
        continue
    fi

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

set +e
pytest -s -v $TEST_FILES \
    --stack-config="$STACK_CONFIG" \
    -k "$PYTEST_PATTERN" \
    --text-model="$TEXT_MODEL" \
    --embedding-model=sentence-transformers/all-MiniLM-L6-v2 \
    --color=yes $EXTRA_PARAMS \
    --capture=tee-sys
exit_code=$?
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

# stop server
if [[ "$STACK_CONFIG" == *"server:"* ]]; then
    echo "Stopping Llama Stack Server..."
    kill $(lsof -i :8321 | awk 'NR>1 {print $2}')
    echo "Llama Stack Server stopped"
fi

echo ""
echo "=== Integration Tests Complete ==="
