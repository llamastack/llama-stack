#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

set -e
cd llama_stack/ui

if [ ! -d node_modules ] || [ ! -x node_modules/.bin/prettier ] || [ ! -x node_modules/.bin/eslint ]; then
  echo "UI dependencies not installed, skipping prettier/linter check"
  exit 0
fi

npm run format
npm run lint
