#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

determine_server_command() {
    local config="$1"
    local server_args=()

    # The env variable will take precedence over the config file
    if [ -n "${ENABLE_POSTGRES_STORE:-}" ]; then
        # TODO: avoid hardcoding the config name
        server_args=("python3" "-m" "llama_stack.core.server.server" "run-with-postgres-store.yaml")
    elif [ -n "$config" ]; then
        server_args=("python3" "-m" "llama_stack.core.server.server" "$config")
    fi
    echo "${server_args[@]}"
}

main() {
    echo "Starting Llama Stack server..."

    local server_command
    server_command=$(determine_server_command "$@")

    if [[ -z "$server_command" ]]; then
        echo "Error: Could not determine server command"
        exit 1
    fi

    printf "Executing: %s\n" "$server_command"
    exec $server_command
}

main "$@"
