#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

LLAMA_STACK_DIR=${LLAMA_STACK_DIR:-}
LLAMA_STACK_CLIENT_DIR=${LLAMA_STACK_CLIENT_DIR:-}

TEST_PYPI_VERSION=${TEST_PYPI_VERSION:-}
PYPI_VERSION=${PYPI_VERSION:-}
BUILD_PLATFORM=${BUILD_PLATFORM:-}
# This timeout (in seconds) is necessary when installing PyTorch via uv since it's likely to time out
# Reference: https://github.com/astral-sh/uv/pull/1694
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-500}

# mounting is not supported by docker buildx, so we use COPY instead
USE_COPY_NOT_MOUNT=${USE_COPY_NOT_MOUNT:-}
# Path to the run.yaml file in the container
RUN_CONFIG_PATH=/app/run.yaml

BUILD_CONTEXT_DIR=$(pwd)

set -euo pipefail

# Define color codes
RED='\033[0;31m'
NC='\033[0m' # No Color

# Usage function
usage() {
  echo "Usage: $0 --image-name <image_name> --container-base <container_base> --normal-deps <pip_dependencies> [--run-config <run_config>] [--external-provider-deps <external_provider_deps>] [--optional-deps <special_pip_deps>]"
  echo "Example: $0 --image-name llama-stack-img --container-base python:3.12-slim --normal-deps 'numpy pandas' --run-config ./run.yaml --external-provider-deps 'foo' --optional-deps 'bar'"
  exit 1
}

# Parse arguments
image_name=""
container_base=""
normal_deps=""
external_provider_deps=""
optional_deps=""
run_config=""
distro_or_config=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --image-name)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --image-name requires a string value" >&2
        usage
      fi
      image_name="$2"
      shift 2
      ;;
    --container-base)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --container-base requires a string value" >&2
        usage
      fi
      container_base="$2"
      shift 2
      ;;
    --normal-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --normal-deps requires a string value" >&2
        usage
      fi
      normal_deps="$2"
      shift 2
      ;;
    --external-provider-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --external-provider-deps requires a string value" >&2
        usage
      fi
      external_provider_deps="$2"
      shift 2
      ;;
    --optional-deps)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --optional-deps requires a string value" >&2
        usage
      fi
      optional_deps="$2"
      shift 2
      ;;
    --run-config)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --run-config requires a string value" >&2
        usage
      fi
      run_config="$2"
      shift 2
      ;;
    --distro-or-config)
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: --distro-or-config requires a string value" >&2
        usage
      fi
      distro_or_config="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

# Check required arguments
if [[ -z "$image_name" || -z "$container_base" || -z "$normal_deps" ]]; then
  echo "Error: --image-name, --container-base, and --normal-deps are required." >&2
  usage
fi

CONTAINER_BINARY=${CONTAINER_BINARY:-docker}
CONTAINER_OPTS=${CONTAINER_OPTS:---progress=plain}
TEMP_DIR=$(mktemp -d)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common.sh"

add_to_container() {
  output_file="$TEMP_DIR/Containerfile"
  if [ -t 0 ]; then
    printf '%s\n' "$1" >>"$output_file"
  else
    cat >>"$output_file"
  fi
}

if ! is_command_available "$CONTAINER_BINARY"; then
  printf "${RED}Error: ${CONTAINER_BINARY} command not found. Is ${CONTAINER_BINARY} installed and in your PATH?${NC}" >&2
  exit 1
fi

if [[ $container_base == *"registry.access.redhat.com/ubi9"* ]]; then
  add_to_container << EOF
FROM $container_base
WORKDIR /app

# We install the Python 3.12 dev headers and build tools so that any
# C-extension wheels (e.g. polyleven, faiss-cpu) can compile successfully.

RUN dnf -y update && dnf install -y iputils git net-tools wget \
    vim-minimal python3.12 python3.12-pip python3.12-wheel \
    python3.12-setuptools python3.12-devel gcc make && \
    ln -s /bin/pip3.12 /bin/pip && ln -s /bin/python3.12 /bin/python && dnf clean all

ENV UV_SYSTEM_PYTHON=1
RUN pip install uv
EOF
else
  add_to_container << EOF
FROM $container_base
WORKDIR /app

RUN apt-get update && apt-get install -y \
       iputils-ping net-tools iproute2 dnsutils telnet \
       curl wget telnet git\
       procps psmisc lsof \
       traceroute \
       bubblewrap \
       gcc \
       && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1
RUN pip install uv
EOF
fi

# Add pip dependencies first since llama-stack is what will change most often
# so we can reuse layers.
if [ -n "$normal_deps" ]; then
  read -ra pip_args <<<  "$normal_deps"
  quoted_deps=$(printf " %q" "${pip_args[@]}")
  add_to_container << EOF
RUN uv pip install --no-cache $quoted_deps
EOF
fi

if [ -n "$optional_deps" ]; then
  IFS='#' read -ra parts <<<"$optional_deps"
  for part in "${parts[@]}"; do
    read -ra pip_args <<< "$part"
    quoted_deps=$(printf " %q" "${pip_args[@]}")
    add_to_container <<EOF
RUN uv pip install --no-cache $quoted_deps
EOF
  done
fi

if [ -n "$external_provider_deps" ]; then
  IFS='#' read -ra parts <<<"$external_provider_deps"
  for part in "${parts[@]}"; do
    read -ra pip_args <<< "$part"
    quoted_deps=$(printf " %q" "${pip_args[@]}")
    add_to_container <<EOF
RUN uv pip install --no-cache $quoted_deps
EOF
    add_to_container <<EOF
RUN python3 - <<PYTHON | uv pip install --no-cache -r -
import importlib
import sys

try:
    package_name = '$part'.split('==')[0].split('>=')[0].split('<=')[0].split('!=')[0].split('<')[0].split('>')[0]
    module = importlib.import_module(f'{package_name}.provider')
    spec = module.get_provider_spec()
    if hasattr(spec, 'pip_packages') and spec.pip_packages:
        if isinstance(spec.pip_packages, (list, tuple)):
            print('\n'.join(spec.pip_packages))
except Exception as e:
    print(f'Error getting provider spec for {package_name}: {e}', file=sys.stderr)
PYTHON
EOF
  done
fi

get_python_cmd() {
    if is_command_available python; then
        echo "python"
    elif is_command_available python3; then
        echo "python3"
    else
        echo "Error: Neither python nor python3 is installed. Please install Python to continue." >&2
        exit 1
    fi
}

if [ -n "$run_config" ]; then
  # Copy the run config to the build context since it's an absolute path
  cp "$run_config" "$BUILD_CONTEXT_DIR/run.yaml"

  # Parse the run.yaml configuration to identify external provider directories
  # If external providers are specified, copy their directory to the container
  # and update the configuration to reference the new container path
  python_cmd=$(get_python_cmd)
  external_providers_dir=$($python_cmd -c "import yaml; config = yaml.safe_load(open('$run_config')); print(config.get('external_providers_dir') or '')")
  external_providers_dir=$(eval echo "$external_providers_dir")
  if [ -n "$external_providers_dir" ]; then
    if [ -d "$external_providers_dir" ]; then
    echo "Copying external providers directory: $external_providers_dir"
    cp -r "$external_providers_dir" "$BUILD_CONTEXT_DIR/providers.d"
    add_to_container << EOF
COPY providers.d /.llama/providers.d
EOF
    fi

    # Edit the run.yaml file to change the external_providers_dir to /.llama/providers.d
    if [ "$(uname)" = "Darwin" ]; then
      sed -i.bak -e 's|external_providers_dir:.*|external_providers_dir: /.llama/providers.d|' "$BUILD_CONTEXT_DIR/run.yaml"
      rm -f "$BUILD_CONTEXT_DIR/run.yaml.bak"
    else
      sed -i 's|external_providers_dir:.*|external_providers_dir: /.llama/providers.d|' "$BUILD_CONTEXT_DIR/run.yaml"
    fi
  fi

  # Copy run config into docker image
  add_to_container << EOF
COPY run.yaml $RUN_CONFIG_PATH
EOF
fi

stack_mount="/app/llama-stack-source"
client_mount="/app/llama-stack-client-source"

install_local_package() {
  local dir="$1"
  local mount_point="$2"
  local name="$3"

  if [ ! -d "$dir" ]; then
    echo "${RED}Warning: $name is set but directory does not exist: $dir${NC}" >&2
    exit 1
  fi

  if [ "$USE_COPY_NOT_MOUNT" = "true" ]; then
    add_to_container << EOF
COPY $dir $mount_point
EOF
  fi
  add_to_container << EOF
RUN uv pip install --no-cache -e $mount_point
EOF
}


if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
  install_local_package "$LLAMA_STACK_CLIENT_DIR" "$client_mount" "LLAMA_STACK_CLIENT_DIR"
fi

if [ -n "$LLAMA_STACK_DIR" ]; then
  install_local_package "$LLAMA_STACK_DIR" "$stack_mount" "LLAMA_STACK_DIR"
else
  if [ -n "$TEST_PYPI_VERSION" ]; then
    # these packages are damaged in test-pypi, so install them first
    add_to_container << EOF
RUN uv pip install --no-cache fastapi libcst
EOF
    add_to_container << EOF
RUN uv pip install --no-cache --extra-index-url https://test.pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  llama-stack==$TEST_PYPI_VERSION

EOF
  else
    if [ -n "$PYPI_VERSION" ]; then
      SPEC_VERSION="llama-stack==${PYPI_VERSION}"
    else
      SPEC_VERSION="llama-stack"
    fi
    add_to_container << EOF
RUN uv pip install --no-cache $SPEC_VERSION
EOF
  fi
fi

# remove uv after installation
  add_to_container << EOF
RUN pip uninstall -y uv
EOF

# If a run config is provided, we use the --config flag
if [[ -n "$run_config" ]]; then
  add_to_container << EOF
ENTRYPOINT ["python", "-m", "llama_stack.core.server.server", "$RUN_CONFIG_PATH"]
EOF
elif [[ "$distro_or_config" != *.yaml ]]; then
  add_to_container << EOF
ENTRYPOINT ["python", "-m", "llama_stack.core.server.server", "$distro_or_config"]
EOF
fi

# Add other require item commands genearic to all containers
add_to_container << EOF

RUN mkdir -p /.llama /.cache && chmod -R g+rw /app /.llama /.cache
EOF

printf "Containerfile created successfully in %s/Containerfile\n\n" "$TEMP_DIR"
cat "$TEMP_DIR"/Containerfile
printf "\n"

# Start building the CLI arguments
CLI_ARGS=()

# Read CONTAINER_OPTS and put it in an array
read -ra CLI_ARGS <<< "$CONTAINER_OPTS"

if [ "$USE_COPY_NOT_MOUNT" != "true" ]; then
  if [ -n "$LLAMA_STACK_DIR" ]; then
    CLI_ARGS+=("-v" "$(readlink -f "$LLAMA_STACK_DIR"):$stack_mount")
  fi
  if [ -n "$LLAMA_STACK_CLIENT_DIR" ]; then
    CLI_ARGS+=("-v" "$(readlink -f "$LLAMA_STACK_CLIENT_DIR"):$client_mount")
  fi
fi

if is_command_available selinuxenabled && selinuxenabled; then
  # Disable SELinux labels -- we don't want to relabel the llama-stack source dir
  CLI_ARGS+=("--security-opt" "label=disable")
fi

# Set version tag based on PyPI version
if [ -n "$PYPI_VERSION" ]; then
  version_tag="$PYPI_VERSION"
elif [ -n "$TEST_PYPI_VERSION" ]; then
  version_tag="test-$TEST_PYPI_VERSION"
elif [[ -n "$LLAMA_STACK_DIR" || -n "$LLAMA_STACK_CLIENT_DIR" ]]; then
  version_tag="dev"
else
  URL="https://pypi.org/pypi/llama-stack/json"
  version_tag=$(curl -s $URL | jq -r '.info.version')
fi

# Add version tag to image name
image_tag="$image_name:$version_tag"

# Detect platform architecture
ARCH=$(uname -m)
if [ -n "$BUILD_PLATFORM" ]; then
  CLI_ARGS+=("--platform" "$BUILD_PLATFORM")
elif [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
  CLI_ARGS+=("--platform" "linux/arm64")
elif [ "$ARCH" = "x86_64" ]; then
  CLI_ARGS+=("--platform" "linux/amd64")
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

echo "PWD: $(pwd)"
echo "Containerfile: $TEMP_DIR/Containerfile"
set -x

$CONTAINER_BINARY build \
  "${CLI_ARGS[@]}" \
  -t "$image_tag" \
  -f "$TEMP_DIR/Containerfile" \
  "$BUILD_CONTEXT_DIR"

# clean up tmp/configs
rm -rf "$BUILD_CONTEXT_DIR/run.yaml" "$TEMP_DIR"
set +x

echo "Success!"
