#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pre-commit hook to enforce a maximum line count per Python file.

Prevents files from growing beyond 1000 lines. Generated files and
vendored specs are excluded.
"""

import sys
from pathlib import Path

MAX_LINES = 1000

EXCLUDE_PATTERNS = [
    "docs/static/",
    "client-sdks/stainless/",
    "tests/integration/recordings/",
]

# Pre-existing large files that haven't been split yet.
# Remove entries from this list as files get refactored.
GRANDFATHERED_FILES = {
    "src/llama_stack/providers/inline/responses/builtin/responses/openai_responses.py",
    "src/llama_stack/providers/inline/responses/builtin/responses/streaming.py",
    "src/llama_stack/providers/inline/scoring/basic/utils/ifeval_utils.py",
    "src/llama_stack/providers/utils/memory/openai_vector_store_mixin.py",
    "src/llama_stack/providers/registry/vector_io.py",
    "src/llama_stack/testing/api_recorder.py",
    "src/llama_stack_api/__init__.py",
    "src/llama_stack_api/openai_responses.py",
    "src/llama_stack_api/inference/models.py",
    "tests/integration/vector_io/test_openai_vector_stores.py",
    "tests/integration/responses/test_openai_responses.py",
    "tests/integration/responses/test_tool_responses.py",
    "tests/unit/providers/responses/builtin/test_openai_responses.py",
    "tests/unit/providers/utils/inference/test_openai_mixin.py",
}


def main() -> int:
    failures: list[tuple[str, int]] = []

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.suffix == ".py":
            continue
        if any(pattern in path_str for pattern in EXCLUDE_PATTERNS):
            continue
        if path_str in GRANDFATHERED_FILES:
            continue
        try:
            line_count = sum(1 for _ in path.open())
        except OSError:
            continue
        if line_count > MAX_LINES:
            failures.append((path_str, line_count))

    if failures:
        for path_str, count in sorted(failures):
            print(f"::error file={path_str}::{path_str} exceeds {MAX_LINES} lines ({count} lines)")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
