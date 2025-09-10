#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Small helper script to extract environment variables from a test setup.
Used by integration-tests.sh to set environment variables before starting the server.
"""

import argparse
import sys

from tests.integration.suites import SETUP_DEFINITIONS, SUITE_DEFINITIONS


def get_setup_env_vars(setup_name, suite_name=None):
    """
    Get environment variables for a setup, with optional suite default fallback.

    Args:
        setup_name: Name of the setup (e.g., 'ollama', 'gpt')
        suite_name: Optional suite name to get default setup if setup_name is None

    Returns:
        Dictionary of environment variables
    """
    # If no setup specified, try to get default from suite
    if not setup_name and suite_name:
        suite = SUITE_DEFINITIONS.get(suite_name)
        if suite and suite.default_setup:
            setup_name = suite.default_setup

    if not setup_name:
        return {}

    setup = SETUP_DEFINITIONS.get(setup_name)
    if not setup:
        print(
            f"Error: Unknown setup '{setup_name}'. Available: {', '.join(sorted(SETUP_DEFINITIONS.keys()))}",
            file=sys.stderr,
        )
        sys.exit(1)

    return setup.env


def main():
    parser = argparse.ArgumentParser(description="Extract environment variables from a test setup")
    parser.add_argument("--setup", help="Setup name (e.g., ollama, gpt)")
    parser.add_argument("--suite", help="Suite name to get default setup from if --setup not provided")
    parser.add_argument("--format", choices=["bash", "json"], default="bash", help="Output format (default: bash)")

    args = parser.parse_args()

    env_vars = get_setup_env_vars(args.setup, args.suite)

    if args.format == "bash":
        # Output as bash export statements
        for key, value in env_vars.items():
            print(f"export {key}='{value}'")
    elif args.format == "json":
        import json

        print(json.dumps(env_vars))


if __name__ == "__main__":
    main()
