#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Check API spec for breaking changes using oasdiff
# This script performs two comparisons:
# 1. Staged API spec vs HEAD - detect breaking changes in the API itself
# 2. API spec vs OpenAI spec - ensure compatibility with OpenAI endpoints (warning only)

set -euo pipefail

OPENAI_SPEC="docs/static/openai-spec-2.3.0.yml"
LLAMA_STACK_SPEC="docs/static/llama-stack-spec.yaml"

# Determine which spec to check (prefer stable, fall back to monolithic)
if [ -f "docs/static/stable-llama-stack-spec.yaml" ]; then
    SPEC="docs/static/stable-llama-stack-spec.yaml"
elif [ -f "$LLAMA_STACK_SPEC" ]; then
    SPEC="$LLAMA_STACK_SPEC"
else
    echo "No API spec found"
    exit 0
fi

# Get the HEAD version for comparison
BASE_SPEC=$(mktemp)
trap "rm -f $BASE_SPEC" EXIT
if ! git show HEAD:"$SPEC" > "$BASE_SPEC" 2>/dev/null; then
    echo "No previous version of $SPEC in HEAD - skipping comparison"
    exit 0
fi

# Check for breaking changes against HEAD
echo "Checking for breaking changes against HEAD..."
oasdiff breaking --fail-on ERR "$BASE_SPEC" "$SPEC" --match-path '^/v1/'

# Check for breaking changes against OpenAI spec (TODO: change from warning to error when GHA check changes)
echo "Checking compatibility with OpenAI spec..."
if ! oasdiff breaking --fail-on ERR "$LLAMA_STACK_SPEC" "$OPENAI_SPEC" --strip-prefix-base "/v1"; then
    echo "Warning: OpenAI compatibility check found issues (non-blocking)"
fi
