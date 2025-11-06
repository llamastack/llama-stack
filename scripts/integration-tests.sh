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
COLLECT_ONLY=false

# Function to display usage
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
    --stack-config STRING    Stack configuration to use (required)
    --suite STRING           Test suite to run (default: 'base')
    --setup STRING           Test setup (models, env) to use (e.g., 'ollama', 'ollama-vision', 'gpt', 'vllm')
    --inference-mode STRING  Inference mode: replay, record-if-missing or record (default: replay)
    --subdirs STRING         Comma-separated list of test subdirectories to run (overrides suite)
    --pattern STRING         Regex pattern to pass to pytest -k
    --collect-only           Collect tests only without running them (skips server startup)
    --help                   Show this help message

Suites are defined in tests/integration/suites.py and define which tests to run.
Setups are defined in tests/integration/setups.py and provide global configuration (models, env).

You can also specify subdirectories (of tests/integration) to select tests from, which will override the suite.

Examples:
    # Basic inference tests with ollama (server mode)
    $0 --stack-config server:ci-tests --suite base --setup ollama

    # Basic inference tests with docker
    $0 --stack-config docker:ci-tests --suite base --setup ollama

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
    --collect-only)
        COLLECT_ONLY=true
        shift
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
if [[ -z "$STACK_CONFIG" && "$COLLECT_ONLY" == false ]]; then
    echo "Error: --stack-config is required"
    usage
    exit 1
fi

if [[ -z "$TEST_SETUP" && -n "$TEST_SUBDIRS" && "$COLLECT_ONLY" == false ]]; then
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

# Set environment variables
export LLAMA_STACK_CLIENT_TIMEOUT=300

THIS_DIR=$(dirname "$0")

if [[ -n "$TEST_SETUP" ]]; then
    EXTRA_PARAMS="--setup=$TEST_SETUP"
fi

if [[ "$COLLECT_ONLY" == true ]]; then
    EXTRA_PARAMS="$EXTRA_PARAMS --collect-only"
fi

# Apply setup-specific environment variables (needed for server startup and tests)
echo "=== Applying Setup Environment Variables ==="

# the server needs this
export LLAMA_STACK_TEST_INFERENCE_MODE="$INFERENCE_MODE"
export SQLITE_STORE_DIR=$(mktemp -d)
echo "Setting SQLITE_STORE_DIR: $SQLITE_STORE_DIR"

# Determine stack config type for api_recorder test isolation
if [[ "$COLLECT_ONLY" == false ]]; then
    if [[ "$STACK_CONFIG" == server:* ]] || [[ "$STACK_CONFIG" == docker:* ]]; then
        export LLAMA_STACK_TEST_STACK_CONFIG_TYPE="server"
        echo "Setting stack config type: server"
    else
        export LLAMA_STACK_TEST_STACK_CONFIG_TYPE="library_client"
        echo "Setting stack config type: library_client"
    fi
fi

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
if [[ "$COLLECT_ONLY" == false ]] && ! command -v llama &>/dev/null; then
    echo "llama could not be found, ensure llama-stack is installed"
    exit 1
fi

if ! command -v pytest &>/dev/null; then
    echo "pytest could not be found, ensure pytest is installed"
    exit 1
fi

# Helper function to find next available port
find_available_port() {
    local start_port=$1
    local port=$start_port
    for ((i=0; i<100; i++)); do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo $port
            return 0
        fi
        ((port++))
    done
    echo "Failed to find available port starting from $start_port" >&2
    return 1
}

# Start Llama Stack Server if needed
if [[ "$STACK_CONFIG" == *"server:"* && "$COLLECT_ONLY" == false ]]; then
    # Find an available port for the server
    LLAMA_STACK_PORT=$(find_available_port 8321)
    if [[ $? -ne 0 ]]; then
        echo "Error: $LLAMA_STACK_PORT"
        exit 1
    fi
    export LLAMA_STACK_PORT
    echo "Will use port: $LLAMA_STACK_PORT"

    stop_server() {
        echo "Stopping Llama Stack Server..."
        pids=$(lsof -i :$LLAMA_STACK_PORT | awk 'NR>1 {print $2}')
        if [[ -n "$pids" ]]; then
            echo "Killing Llama Stack Server processes: $pids"
            kill -9 $pids
        else
            echo "No Llama Stack Server processes found ?!"
        fi
        echo "Llama Stack Server stopped"
    }

    echo "=== Starting Llama Stack Server ==="
    export LLAMA_STACK_LOG_WIDTH=120

        # Configure telemetry collector for server mode
        # Use a fixed port for the OTEL collector so the server can connect to it
        COLLECTOR_PORT=4317
        export LLAMA_STACK_TEST_COLLECTOR_PORT="${COLLECTOR_PORT}"
        # Disabled: https://github.com/llamastack/llama-stack/issues/4089
        #export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:${COLLECTOR_PORT}"
        export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
        export OTEL_BSP_SCHEDULE_DELAY="200"
        export OTEL_BSP_EXPORT_TIMEOUT="2000"
        export OTEL_METRIC_EXPORT_INTERVAL="200"

    # remove "server:" from STACK_CONFIG
    stack_config=$(echo "$STACK_CONFIG" | sed 's/^server://')
    nohup llama stack run $stack_config >server.log 2>&1 &

    echo "Waiting for Llama Stack Server to start on port $LLAMA_STACK_PORT..."
    for i in {1..30}; do
        if curl -s http://localhost:$LLAMA_STACK_PORT/v1/health 2>/dev/null | grep -q "OK"; then
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

    trap stop_server EXIT ERR INT TERM
fi

# Start Docker Container if needed
if [[ "$STACK_CONFIG" == *"docker:"* && "$COLLECT_ONLY" == false ]]; then
    stop_container() {
        echo "Stopping Docker container..."
        container_name="llama-stack-test-$DISTRO"
        if docker ps -a --format '{{.Names}}' | grep -q "^${container_name}$"; then
            echo "Dumping container logs before stopping..."
            docker logs "$container_name" >"docker-${DISTRO}-${INFERENCE_MODE}.log" 2>&1 || true
            echo "Stopping and removing container: $container_name"
            docker stop "$container_name" 2>/dev/null || true
            docker rm "$container_name" 2>/dev/null || true
        else
            echo "No container named $container_name found"
        fi
        echo "Docker container stopped"
    }

    # Extract distribution name from docker:distro format
    DISTRO=$(echo "$STACK_CONFIG" | sed 's/^docker://')
    # Find an available port for the docker container
    LLAMA_STACK_PORT=$(find_available_port 8321)
    if [[ $? -ne 0 ]]; then
        echo "Error: $LLAMA_STACK_PORT"
        exit 1
    fi
    export LLAMA_STACK_PORT
    echo "Will use port: $LLAMA_STACK_PORT"

    echo "=== Building Docker Image for distribution: $DISTRO ==="
    containerfile="$ROOT_DIR/containers/Containerfile"
    if [[ ! -f "$containerfile" ]]; then
        echo "❌ Containerfile not found at $containerfile"
        exit 1
    fi

    build_cmd=(
        docker
        build
        "$ROOT_DIR"
        -f "$containerfile"
        --tag "localhost/distribution-$DISTRO:dev"
        --build-arg "DISTRO_NAME=$DISTRO"
        --build-arg "INSTALL_MODE=editable"
        --build-arg "LLAMA_STACK_DIR=/workspace"
    )

    # Pass UV index configuration for release branches
    if [[ -n "${UV_EXTRA_INDEX_URL:-}" ]]; then
        echo "Adding UV_EXTRA_INDEX_URL to docker build: $UV_EXTRA_INDEX_URL"
        build_cmd+=(--build-arg "UV_EXTRA_INDEX_URL=$UV_EXTRA_INDEX_URL")
    fi
    if [[ -n "${UV_INDEX_STRATEGY:-}" ]]; then
        echo "Adding UV_INDEX_STRATEGY to docker build: $UV_INDEX_STRATEGY"
        build_cmd+=(--build-arg "UV_INDEX_STRATEGY=$UV_INDEX_STRATEGY")
    fi

    if ! "${build_cmd[@]}"; then
        echo "❌ Failed to build Docker image"
        exit 1
    fi

    echo ""
    echo "=== Starting Docker Container ==="
    container_name="llama-stack-test-$DISTRO"

    # Stop and remove existing container if it exists
    docker stop "$container_name" 2>/dev/null || true
    docker rm "$container_name" 2>/dev/null || true

    # Configure telemetry collector port shared between host and container
    COLLECTOR_PORT=4317
    export LLAMA_STACK_TEST_COLLECTOR_PORT="${COLLECTOR_PORT}"

    # Build environment variables for docker run
    DOCKER_ENV_VARS=""
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e LLAMA_STACK_TEST_INFERENCE_MODE=$INFERENCE_MODE"
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e LLAMA_STACK_TEST_STACK_CONFIG_TYPE=server"
    # Disabled: https://github.com/llamastack/llama-stack/issues/4089
    #DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:${COLLECTOR_PORT}"
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OTEL_METRIC_EXPORT_INTERVAL=200"
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OTEL_BSP_SCHEDULE_DELAY=200"
    DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OTEL_BSP_EXPORT_TIMEOUT=2000"

    # Pass through API keys if they exist
    [ -n "${TOGETHER_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e TOGETHER_API_KEY=$TOGETHER_API_KEY"
    [ -n "${FIREWORKS_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e FIREWORKS_API_KEY=$FIREWORKS_API_KEY"
    [ -n "${TAVILY_SEARCH_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e TAVILY_SEARCH_API_KEY=$TAVILY_SEARCH_API_KEY"
    [ -n "${OPENAI_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OPENAI_API_KEY=$OPENAI_API_KEY"
    [ -n "${ANTHROPIC_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    [ -n "${GROQ_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e GROQ_API_KEY=$GROQ_API_KEY"
    [ -n "${GEMINI_API_KEY:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e GEMINI_API_KEY=$GEMINI_API_KEY"
    [ -n "${OLLAMA_URL:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e OLLAMA_URL=$OLLAMA_URL"
    [ -n "${SAFETY_MODEL:-}" ] && DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e SAFETY_MODEL=$SAFETY_MODEL"

    if [[ "$TEST_SETUP" == "vllm" ]]; then
        DOCKER_ENV_VARS="$DOCKER_ENV_VARS -e VLLM_URL=http://localhost:8000/v1"
    fi

    # Determine the actual image name (may have localhost/ prefix)
    IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "distribution-$DISTRO:dev$" | head -1)
    if [[ -z "$IMAGE_NAME" ]]; then
        echo "❌ Error: Could not find image for distribution-$DISTRO:dev"
        exit 1
    fi
    echo "Using image: $IMAGE_NAME"

    # On macOS/Darwin, --network host doesn't work as expected due to Docker running in a VM
    # Use regular port mapping instead
    NETWORK_MODE=""
    PORT_MAPPINGS=""
    if [[ "$(uname)" != "Darwin" ]] && [[ "$(uname)" != *"MINGW"* ]]; then
        NETWORK_MODE="--network host"
    else
        # On non-Linux (macOS, Windows), need explicit port mappings for both app and telemetry
        PORT_MAPPINGS="-p $LLAMA_STACK_PORT:$LLAMA_STACK_PORT -p $COLLECTOR_PORT:$COLLECTOR_PORT"
        echo "Using bridge networking with port mapping (non-Linux)"
    fi

    docker run -d $NETWORK_MODE --name "$container_name" \
        $PORT_MAPPINGS \
        $DOCKER_ENV_VARS \
        "$IMAGE_NAME" \
        --port $LLAMA_STACK_PORT

    echo "Waiting for Docker container to start..."
    for i in {1..30}; do
        if curl -s http://localhost:$LLAMA_STACK_PORT/v1/health 2>/dev/null | grep -q "OK"; then
            echo "✅ Docker container started successfully"
            break
        fi
        if [[ $i -eq 30 ]]; then
            echo "❌ Docker container failed to start"
            echo "Container logs:"
            docker logs "$container_name"
            exit 1
        fi
        sleep 1
    done
    echo ""

    # Update STACK_CONFIG to point to the running container
    STACK_CONFIG="http://localhost:$LLAMA_STACK_PORT"

    trap stop_container EXIT ERR INT TERM
fi

# Run tests
echo "=== Running Integration Tests ==="
EXCLUDE_TESTS="builtin_tool or safety_with_image or code_interpreter or test_rag"

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

STACK_CONFIG_ARG=""
if [[ -n "$STACK_CONFIG" ]]; then
    STACK_CONFIG_ARG="--stack-config=$STACK_CONFIG"
fi

pytest -s -v $PYTEST_TARGET \
    $STACK_CONFIG_ARG \
    --inference-mode="$INFERENCE_MODE" \
    -k "$PYTEST_PATTERN" \
    $EXTRA_PARAMS \
    --color=yes \
    --embedding-model=sentence-transformers/nomic-ai/nomic-embed-text-v1.5 \
    --color=yes $EXTRA_PARAMS \
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
    echo ""
    # Output server or container logs based on stack config
    if [[ "$STACK_CONFIG" == *"server:"* && -f "server.log" ]]; then
        echo "--- Server side failures can be located inside server.log (available from artifacts on CI) ---"
    elif [[ "$STACK_CONFIG" == *"docker:"* ]]; then
        docker_log_file="docker-${DISTRO}-${INFERENCE_MODE}.log"
        if [[ -f "$docker_log_file" ]]; then
            echo "--- Server side failures can be located inside $docker_log_file (available from artifacts on CI) ---"
        fi
    fi

    exit 1
fi

echo ""
echo "=== Integration Tests Complete ==="
