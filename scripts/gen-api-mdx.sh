#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Generates API documentation MDX files from OpenAPI specs.
# These files are pre-generated and committed to avoid slow builds
# in the llamastack.github.io documentation workflow.
#
# This script is run automatically by pre-commit when OpenAPI specs change.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO_ROOT=$(dirname "$THIS_DIR")
DOCS_DIR="$REPO_ROOT/docs"

# Skip in CI environments - the compressed content in generated MDX files
# is non-deterministic across environments (different zlib/Node.js versions),
# so we only regenerate locally and commit the results.
if [ "${SKIP_API_DOCS_GEN:-}" = "1" ] || [ "${CI:-}" = "true" ]; then
    echo "Skipping API docs generation (CI or SKIP_API_DOCS_GEN=1)"
    exit 0
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "Warning: Node.js not found, skipping API docs generation"
    echo "Install Node.js 20+ to generate API documentation"
    exit 0
fi

# Check if npm dependencies are installed
if [ ! -d "$DOCS_DIR/node_modules" ]; then
    echo "Installing npm dependencies in docs/..."
    (cd "$DOCS_DIR" && npm ci --silent)
fi

echo "Generating API documentation from OpenAPI specs..."

# Clean existing generated docs (but preserve index.mdx)
for dir in api api-experimental api-deprecated; do
    target_dir="$DOCS_DIR/docs/$dir"
    if [ -d "$target_dir" ]; then
        # Remove all files except index.mdx and sidebar files
        find "$target_dir" -name "*.mdx" -type f ! -name "index.mdx" -delete 2>/dev/null || true
    fi
done

# Generate the API docs
(cd "$DOCS_DIR" && npm run gen-api-docs all)

# Fix formatting issues from docusaurus-plugin-openapi-docs:
# - Remove trailing whitespace from lines
# - Remove trailing blank lines (match end-of-file-fixer behavior)
echo "Fixing trailing whitespace in generated files..."
for dir in api api-experimental api-deprecated; do
    target_dir="$DOCS_DIR/docs/$dir"
    if [ -d "$target_dir" ]; then
        find "$target_dir" \( -name "*.mdx" -o -name "*.ts" \) -type f | while read -r file; do
            # Remove trailing whitespace from all lines
            sed -i 's/[[:space:]]*$//' "$file"
            # Remove trailing blank lines (keep content, single newline at end)
            # This matches end-of-file-fixer behavior
            printf '%s\n' "$(cat "$file")" > "$file"
        done
    fi
done

# Count generated files for summary
stable_count=$(find "$DOCS_DIR/docs/api" -name "*.mdx" -type f 2>/dev/null | wc -l)
experimental_count=$(find "$DOCS_DIR/docs/api-experimental" -name "*.mdx" -type f 2>/dev/null | wc -l)
deprecated_count=$(find "$DOCS_DIR/docs/api-deprecated" -name "*.mdx" -type f 2>/dev/null | wc -l)

echo "Generated API docs:"
echo "  - Stable API: $stable_count files"
echo "  - Experimental API: $experimental_count files"
echo "  - Deprecated API: $deprecated_count files"
echo "Done!"
