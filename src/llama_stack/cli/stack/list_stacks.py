# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR


class StackListBuilds(Subcommand):
    """List available distributions (both built-in and custom)"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama stack list",
            description="list available distributions",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._list_stack_command)

    def _get_distribution_dirs(self) -> dict[str, tuple[Path, str]]:
        """Return a dictionary of distribution names and their paths with source type

        Returns:
            dict mapping distro name to (path, source_type) where source_type is 'built-in' or 'custom'
        """
        distributions = {}

        # Get built-in distributions from source code
        distro_dir = Path(__file__).parent.parent.parent / "distributions"
        if distro_dir.exists():
            for stack_dir in distro_dir.iterdir():
                if stack_dir.is_dir() and not stack_dir.name.startswith(".") and not stack_dir.name.startswith("__"):
                    distributions[stack_dir.name] = (stack_dir, "built-in")

        # Get custom/run distributions from ~/.llama/distributions
        # These override built-in ones if they have the same name
        if DISTRIBS_BASE_DIR.exists():
            for stack_dir in DISTRIBS_BASE_DIR.iterdir():
                if stack_dir.is_dir() and not stack_dir.name.startswith("."):
                    # Clean up the name (remove llamastack- prefix if present)
                    name = stack_dir.name.replace("llamastack-", "")
                    distributions[name] = (stack_dir, "custom")

        return distributions

    def _list_stack_command(self, args: argparse.Namespace) -> None:
        distributions = self._get_distribution_dirs()

        if not distributions:
            print("No distributions found")
            return

        headers = ["Stack Name", "Source", "Path", "Build Config", "Run Config"]
        rows = []
        for name, (path, source_type) in sorted(distributions.items()):
            row = [name, source_type, str(path)]
            # Check for build and run config files
            # For built-in distributions, configs are named build.yaml and run.yaml
            # For custom distributions, configs are named {name}-build.yaml and {name}-run.yaml
            if source_type == "built-in":
                build_config = "Yes" if (path / "build.yaml").exists() else "No"
                run_config = "Yes" if (path / "run.yaml").exists() else "No"
            else:
                build_config = "Yes" if (path / f"{name}-build.yaml").exists() else "No"
                run_config = "Yes" if (path / f"{name}-run.yaml").exists() else "No"
            row.extend([build_config, run_config])
            rows.append(row)
        print_table(rows, headers, separate_rows=True)
