#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -euo pipefail

# Detect current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# On release branches, use test.pypi as extra index for RC versions
if [[ "$BRANCH" =~ ^release-[0-9]+\.[0-9]+\.x$ ]]; then
  echo "Detected release branch: $BRANCH"
  echo "Setting UV_EXTRA_INDEX_URL=https://test.pypi.org/simple/"
  export UV_EXTRA_INDEX_URL="https://test.pypi.org/simple/"
  export UV_INDEX_STRATEGY="unsafe-best-match"
fi

# Run uv lock
exec uv lock "$@"
