#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Documentation Build Script with Caching
#
# This script builds the documentation with optimized API docs generation.
# It uses hash-based caching to skip regenerating API docs when the OpenAPI
# specs haven't changed.
#
# Usage:
#   ./scripts/build-docs.sh [options]
#
# Options:
#   --force      Force regeneration of API docs even if specs haven't changed
#   --parallel   Generate API docs for multiple specs in parallel
#   --skip-api   Skip API docs generation entirely (use existing docs)
#   --serve      Start a local server after building
#
# Environment Variables:
#   SKIP_API_DOCS=1    Skip API docs generation
#   FORCE_API_DOCS=1   Force API docs regeneration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DOCS_DIR="$REPO_ROOT/docs"

# Parse arguments
FORCE=""
PARALLEL=""
SKIP_API=""
SERVE=""

for arg in "$@"; do
    case $arg in
        --force)
            FORCE="--force"
            ;;
        --parallel)
            PARALLEL="--parallel"
            ;;
        --skip-api)
            SKIP_API="1"
            ;;
        --serve)
            SERVE="1"
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Usage: $0 [--force] [--parallel] [--skip-api] [--serve]"
            exit 1
            ;;
    esac
done

# Check environment variables
if [[ "${SKIP_API_DOCS:-}" == "1" ]]; then
    SKIP_API="1"
fi

if [[ "${FORCE_API_DOCS:-}" == "1" ]]; then
    FORCE="--force"
fi

cd "$DOCS_DIR"

echo "=================================="
echo "Llama Stack Documentation Builder"
echo "=================================="
echo ""

# Install dependencies if needed
if [[ ! -d "node_modules" ]]; then
    echo "Installing dependencies..."
    npm ci
    echo ""
fi

# Generate API docs with caching (unless skipped)
if [[ -z "$SKIP_API" ]]; then
    echo "Generating API documentation..."
    node scripts/gen-api-docs.js all $FORCE $PARALLEL
    echo ""
else
    echo "Skipping API docs generation (--skip-api or SKIP_API_DOCS=1)"
    echo ""
fi

# Sync files
echo "Syncing imported files..."
npm run sync-files
echo ""

# Build the documentation
echo "Building documentation..."
npm run build
echo ""

echo "=================================="
echo "Build complete!"
echo "=================================="
echo "Output: $DOCS_DIR/build/"

# Serve if requested
if [[ -n "$SERVE" ]]; then
    echo ""
    echo "Starting local server..."
    npm run serve
fi
