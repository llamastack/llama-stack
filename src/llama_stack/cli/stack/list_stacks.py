# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table


class StackListBuilds(Subcommand):
    """List built stacks in .llama/distributions directory"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama stack list",
            description="list the build stacks",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._list_stack_command)

    def _get_distribution_dirs(self) -> dict[str, Path]:
        """Return a dictionary of distribution names and their paths"""
        distributions = {}
        dist_dir = Path.home() / ".llama" / "distributions"

        if dist_dir.exists():
            for stack_dir in dist_dir.iterdir():
                if stack_dir.is_dir():
                    distributions[stack_dir.name] = stack_dir
        return distributions

    def _list_stack_command(self, args: argparse.Namespace) -> None:
        distributions = self._get_distribution_dirs()

        if not distributions:
            print("No stacks found in ~/.llama/distributions")
            return

        headers = ["Stack Name", "Path"]
        headers.extend(["Build Config", "Run Config"])
        rows = []
        for name, path in distributions.items():
            row = [name, str(path)]
            # Check for build and run config files
            build_config = "Yes" if (path / f"{name}-build.yaml").exists() else "No"
            run_config = "Yes" if (path / f"{name}-run.yaml").exists() else "No"
            row.extend([build_config, run_config])
            rows.append(row)
        print_table(rows, headers, separate_rows=True)
