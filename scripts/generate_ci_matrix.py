#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Generate CI test matrix from suites.py with schedule/input overrides.

This script is used by .github/workflows/integration-tests.yml to generate
the test matrix dynamically based on the CI_MATRIX definition in suites.py.
"""

import json
import sys
from pathlib import Path

# Add tests/integration to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tests/integration"))

from suites import CI_MATRIX


def generate_matrix(schedule="", test_setup=""):
    """
    Generate test matrix based on schedule or manual input.

    Args:
        schedule: GitHub cron schedule string (e.g., "1 0 * * 0" for weekly)
        test_setup: Manual test setup input (e.g., "ollama-vision")

    Returns:
        Matrix configuration as JSON string
    """
    # Weekly vllm test on Sunday
    if schedule == "1 0 * * 0":
        matrix = [{"suite": "base", "setup": "vllm"}]
    # Manual input for specific setup
    elif test_setup == "ollama-vision":
        matrix = [{"suite": "vision", "setup": "ollama-vision"}]
    # Default: use CI_MATRIX from suites.py
    else:
        matrix = CI_MATRIX

    # GitHub Actions expects {"include": [...]} format
    return json.dumps({"include": matrix})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate CI test matrix")
    parser.add_argument("--schedule", default="", help="GitHub schedule cron string")
    parser.add_argument("--test-setup", default="", help="Manual test setup input")

    args = parser.parse_args()

    print(generate_matrix(args.schedule, args.test_setup))
