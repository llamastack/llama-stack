#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

[ -z "${BASH_VERSION:-}" ] && exec /usr/bin/env bash "$0" "$@"
if set -o | grep -Eq 'posix[[:space:]]+on'; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TELEMETRY_DIR="${SCRIPT_DIR}/telemetry"
TELEMETRY_REMOTE_BASE_DEFAULT="https://raw.githubusercontent.com/llamastack/llama-stack/main/scripts/telemetry"

PORT=8321
OLLAMA_PORT=11434
MODEL_ALIAS="llama3.2:3b"
SERVER_IMAGE="docker.io/llamastack/distribution-starter:latest"
WAIT_TIMEOUT=30
TEMP_LOG=""
WITH_TELEMETRY=true
TELEMETRY_SERVICE_NAME="llama-stack"
TELEMETRY_SINKS="otel_trace,otel_metric"
OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4318"
TEMP_TELEMETRY_DIR=""

# Cleanup function to remove temporary files
cleanup() {
  if [ -n "$TEMP_LOG" ] && [ -f "$TEMP_LOG" ]; then
    rm -f "$TEMP_LOG"
  fi
  if [ -n "$TEMP_TELEMETRY_DIR" ] && [ -d "$TEMP_TELEMETRY_DIR" ]; then
    rm -rf "$TEMP_TELEMETRY_DIR"
  fi
}

# Set up trap to clean up on exit, error, or interrupt
trap cleanup EXIT ERR INT TERM

log(){ printf "\e[1;32m%s\e[0m\n" "$*"; }
die(){
  printf "\e[1;31m❌ %s\e[0m\n" "$*" >&2
  printf "\e[1;31m🐛 Report an issue @ https://github.com/llamastack/llama-stack/issues if you think it's a bug\e[0m\n" >&2
  exit 1
}

# Helper function to execute command with logging
execute_with_log() {
  local cmd=("$@")
  TEMP_LOG=$(mktemp)
  if ! "${cmd[@]}" > "$TEMP_LOG" 2>&1; then
    log "❌ Command failed; dumping output:"
    log "Command that failed: ${cmd[*]}"
    log "Command output:"
    cat "$TEMP_LOG"
    return 1
  fi
  return 0
}

wait_for_service() {
  local url="$1"
  local pattern="$2"
  local timeout="$3"
  local name="$4"
  local start ts
  log "⏳ Waiting for ${name}..."
  start=$(date +%s)
  while true; do
    if curl --retry 5 --retry-delay 1 --retry-max-time "$timeout" --retry-all-errors --silent --fail "$url" 2>/dev/null | grep -q "$pattern"; then
      break
    fi
    ts=$(date +%s)
    if (( ts - start >= timeout )); then
      return 1
    fi
    printf '.'
  done
  printf '\n'
  return 0
}

usage() {
    cat << EOF
📚 Llama Stack Deployment Script

Description:
    This script sets up and deploys Llama Stack with Ollama integration in containers.
    It handles both Docker and Podman runtimes and includes automatic platform detection.

Usage:
    $(basename "$0") [OPTIONS]

Options:
    -p, --port PORT            Server port for Llama Stack (default: ${PORT})
    -o, --ollama-port PORT     Ollama service port (default: ${OLLAMA_PORT})
    -m, --model MODEL          Model alias to use (default: ${MODEL_ALIAS})
    -i, --image IMAGE          Server image (default: ${SERVER_IMAGE})
    -t, --timeout SECONDS      Service wait timeout in seconds (default: ${WAIT_TIMEOUT})
    --with-telemetry           Provision Jaeger, OTEL Collector, Prometheus, and Grafana (default: enabled)
    --no-telemetry, --without-telemetry
                              Skip provisioning the telemetry stack
    --telemetry-service NAME   Service name reported to telemetry (default: ${TELEMETRY_SERVICE_NAME})
    --telemetry-sinks SINKS    Comma-separated telemetry sinks (default: ${TELEMETRY_SINKS})
    --otel-endpoint URL        OTLP endpoint provided to Llama Stack (default: ${OTEL_EXPORTER_OTLP_ENDPOINT})
    -h, --help                 Show this help message

For more information:
    Documentation: https://llamastack.github.io/latest/
    GitHub: https://github.com/llamastack/llama-stack

Report issues:
    https://github.com/llamastack/llama-stack/issues
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -o|--ollama-port)
            OLLAMA_PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_ALIAS="$2"
            shift 2
            ;;
        -i|--image)
            SERVER_IMAGE="$2"
            shift 2
            ;;
        -t|--timeout)
            WAIT_TIMEOUT="$2"
            shift 2
            ;;
        --with-telemetry)
            WITH_TELEMETRY=true
            shift
            ;;
        --no-telemetry|--without-telemetry)
            WITH_TELEMETRY=false
            shift
            ;;
        --telemetry-service)
            TELEMETRY_SERVICE_NAME="$2"
            shift 2
            ;;
        --telemetry-sinks)
            TELEMETRY_SINKS="$2"
            shift 2
            ;;
        --otel-endpoint)
            OTEL_EXPORTER_OTLP_ENDPOINT="$2"
            shift 2
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

if command -v docker &> /dev/null; then
  ENGINE="docker"
elif command -v podman &> /dev/null; then
  ENGINE="podman"
else
  die "Docker or Podman is required. Install Docker: https://docs.docker.com/get-docker/ or Podman: https://podman.io/getting-started/installation"
fi

# Explicitly set the platform for the host architecture
HOST_ARCH="$(uname -m)"
if [ "$HOST_ARCH" = "arm64" ]; then
  if [ "$ENGINE" = "docker" ]; then
    PLATFORM_OPTS=( --platform linux/amd64 )
  else
    PLATFORM_OPTS=( --os linux --arch amd64 )
  fi
else
  PLATFORM_OPTS=()
fi

# macOS + Podman: ensure VM is running before we try to launch containers
# If you need GPU passthrough under Podman on macOS, init the VM with libkrun:
#   CONTAINERS_MACHINE_PROVIDER=libkrun podman machine init
if [ "$ENGINE" = "podman" ] && [ "$(uname -s)" = "Darwin" ]; then
  if ! podman info &>/dev/null; then
    log "⌛️ Initializing Podman VM..."
    podman machine init &>/dev/null || true
    podman machine start &>/dev/null || true

    log "⌛️ Waiting for Podman API..."
    until podman info &>/dev/null; do
      sleep 1
    done
    log "✅ Podman VM is up."
  fi
fi

# Clean up any leftovers from earlier runs
containers=(ollama-server llama-stack)
if [ "$WITH_TELEMETRY" = true ]; then
  containers+=(jaeger otel-collector prometheus grafana)
fi
for name in "${containers[@]}"; do
  ids=$($ENGINE ps -aq --filter "name=^${name}$")
  if [ -n "$ids" ]; then
    log "⚠️  Found existing container(s) for '${name}', removing..."
    if ! execute_with_log $ENGINE rm -f "$ids"; then
      die "Container cleanup failed"
    fi
  fi
done

###############################################################################
# 0. Create a shared network
###############################################################################
if ! $ENGINE network inspect llama-net >/dev/null 2>&1; then
  log "🌐 Creating network..."
  if ! execute_with_log $ENGINE network create llama-net; then
    die "Network creation failed"
  fi
fi

###############################################################################
# Telemetry Stack
###############################################################################
if [ "$WITH_TELEMETRY" = true ]; then
  TELEMETRY_ASSETS_DIR="$TELEMETRY_DIR"
  if [ ! -d "$TELEMETRY_ASSETS_DIR" ]; then
    TELEMETRY_REMOTE_BASE="${TELEMETRY_REMOTE_BASE:-$TELEMETRY_REMOTE_BASE_DEFAULT}"
    TEMP_TELEMETRY_DIR="$(mktemp -d)"
    TELEMETRY_ASSETS_DIR="$TEMP_TELEMETRY_DIR"
    log "📥 Fetching telemetry assets from ${TELEMETRY_REMOTE_BASE}..."
    for asset in otel-collector-config.yaml prometheus.yml grafana-datasources.yaml; do
      if ! curl -fsSL "${TELEMETRY_REMOTE_BASE}/${asset}" -o "${TELEMETRY_ASSETS_DIR}/${asset}"; then
        die "Failed to download telemetry asset: ${asset}"
      fi
    done
  fi

  log "📡 Starting telemetry stack..."

  if ! execute_with_log $ENGINE run -d "${PLATFORM_OPTS[@]}" --name jaeger \
    --network llama-net \
    -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
    -p 16686:16686 \
    -p 14250:14250 \
    -p 9411:9411 \
    docker.io/jaegertracing/all-in-one:latest > /dev/null 2>&1; then
    die "Jaeger startup failed"
  fi

  if ! execute_with_log $ENGINE run -d "${PLATFORM_OPTS[@]}" --name otel-collector \
    --network llama-net \
    -p 4318:4318 \
    -p 4317:4317 \
    -p 9464:9464 \
    -p 13133:13133 \
    -v "${TELEMETRY_ASSETS_DIR}/otel-collector-config.yaml:/etc/otel-collector-config.yaml:Z" \
    docker.io/otel/opentelemetry-collector-contrib:latest \
    --config /etc/otel-collector-config.yaml > /dev/null 2>&1; then
    die "OpenTelemetry Collector startup failed"
  fi

  if ! execute_with_log $ENGINE run -d "${PLATFORM_OPTS[@]}" --name prometheus \
    --network llama-net \
    -p 9090:9090 \
    -v "${TELEMETRY_ASSETS_DIR}/prometheus.yml:/etc/prometheus/prometheus.yml:Z" \
    docker.io/prom/prometheus:latest \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.console.templates=/etc/prometheus/consoles \
    --storage.tsdb.retention.time=200h \
    --web.enable-lifecycle > /dev/null 2>&1; then
    die "Prometheus startup failed"
  fi

  if ! execute_with_log $ENGINE run -d "${PLATFORM_OPTS[@]}" --name grafana \
    --network llama-net \
    -p 3000:3000 \
    -e GF_SECURITY_ADMIN_PASSWORD=admin \
    -e GF_USERS_ALLOW_SIGN_UP=false \
    -v "${TELEMETRY_ASSETS_DIR}/grafana-datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml:Z" \
    docker.io/grafana/grafana:11.0.0 > /dev/null 2>&1; then
    die "Grafana startup failed"
  fi
fi

###############################################################################
# 1. Ollama
###############################################################################
log "🦙 Starting Ollama..."
if ! execute_with_log $ENGINE run -d "${PLATFORM_OPTS[@]}" --name ollama-server \
  --network llama-net \
  -p "${OLLAMA_PORT}:${OLLAMA_PORT}" \
  docker.io/ollama/ollama > /dev/null 2>&1; then
  die "Ollama startup failed"
fi

if ! wait_for_service "http://localhost:${OLLAMA_PORT}/" "Ollama" "$WAIT_TIMEOUT" "Ollama daemon"; then
  log "❌ Ollama daemon did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs --tail 200 ollama-server
  die "Ollama startup failed"
fi

log "📦 Ensuring model is pulled: ${MODEL_ALIAS}..."
if ! execute_with_log $ENGINE exec ollama-server ollama pull "${MODEL_ALIAS}"; then
  log "❌ Failed to pull model ${MODEL_ALIAS}; dumping container logs:"
  $ENGINE logs --tail 200 ollama-server
  die "Model pull failed"
fi

###############################################################################
# 2. Llama‑Stack
###############################################################################
server_env_opts=()
if [ "$WITH_TELEMETRY" = true ]; then
  server_env_opts+=(
    -e TELEMETRY_SINKS="${TELEMETRY_SINKS}"
    -e OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT}"
    -e OTEL_SERVICE_NAME="${TELEMETRY_SERVICE_NAME}"
  )
fi

cmd=( run -d "${PLATFORM_OPTS[@]}" --name llama-stack \
      --network llama-net \
      -p "${PORT}:${PORT}" \
      "${server_env_opts[@]}" \
      -e OLLAMA_URL="http://ollama-server:${OLLAMA_PORT}" \
      "${SERVER_IMAGE}" --port "${PORT}")

log "🦙 Starting Llama Stack..."
if ! execute_with_log $ENGINE "${cmd[@]}"; then
  die "Llama Stack startup failed"
fi

if ! wait_for_service "http://127.0.0.1:${PORT}/v1/health" "OK" "$WAIT_TIMEOUT" "Llama Stack API"; then
  log "❌ Llama Stack did not become ready in ${WAIT_TIMEOUT}s; dumping container logs:"
  $ENGINE logs --tail 200 llama-stack
  die "Llama Stack startup failed"
fi

###############################################################################
# Done
###############################################################################
log ""
log "🎉 Llama Stack is ready!"
log "👉  API endpoint: http://localhost:${PORT}"
log "📖 Documentation: https://llamastack.github.io/latest/references/api_reference/index.html"
log "💻 To access the llama stack CLI, exec into the container:"
log "   $ENGINE exec -ti llama-stack bash"
if [ "$WITH_TELEMETRY" = true ]; then
  log "📡 Telemetry dashboards:"
  log "   Jaeger UI:      http://localhost:16686"
  log "   Prometheus UI:  http://localhost:9090"
  log "   Grafana UI:     http://localhost:3000 (admin/admin)"
  log "   OTEL Collector: http://localhost:4318"
fi
log "🐛 Report an issue @ https://github.com/llamastack/llama-stack/issues if you think it's a bug"
log ""
