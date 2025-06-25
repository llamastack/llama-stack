#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rich.console import Console
from rich.table import Table

from llama_stack.cli.subcommand import Subcommand

# Import build info at module level for testing
try:
    from .build_info import BUILD_INFO
except ImportError:
    BUILD_INFO = None  # type: ignore


def print_simple_table(rows, width=80):
    """Print a simple table with fixed width"""

    table = Table(show_header=True, width=width)
    table.add_column("Property", width=30)
    table.add_column("Value", width=46)

    for row in rows:
        table.add_row(*row)

    Console(width=width).print(table)


class VersionCommand(Subcommand):
    """Display version information for Llama Stack CLI and server"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "version",
            prog="llama version",
            description="Display version information for Llama Stack CLI and server",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_version_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "-o",
            "--output",
            choices=["table", "json"],
            default="table",
            help="Output format (table, json)",
        )
        self.parser.add_argument(
            "-b",
            "--build-info",
            action="store_true",
            help="Include build information (git commit, date, branch, tag)",
        )
        self.parser.add_argument(
            "-d",
            "--dependencies",
            action="store_true",
            help="Include dependency versions information",
        )
        self.parser.add_argument(
            "-a",
            "--all",
            action="store_true",
            help="Display all information (build info + dependencies)",
        )

    def _get_package_version(self, package_name: str) -> str:
        """Get version of a package, return 'unknown' if not found"""
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "unknown"

    def _get_build_info(self) -> dict:
        """Get build information from build_info.py"""
        build_info = {
            "commit_hash": "unknown",
            "commit_date": "unknown",
            "branch": "unknown",
            "tag": "unknown",
            "build_timestamp": "unknown",
        }

        try:
            if BUILD_INFO is not None:
                build_info.update(
                    {
                        "commit_hash": BUILD_INFO.get("git_commit", "unknown"),
                        "commit_date": BUILD_INFO.get("git_commit_date", "unknown"),
                        "branch": BUILD_INFO.get("git_branch", "unknown"),
                        "tag": BUILD_INFO.get("git_tag", "unknown"),
                        "build_timestamp": BUILD_INFO.get("build_timestamp", "unknown"),
                    }
                )
        except (ImportError, AttributeError, TypeError):
            # build_info.py not available or BUILD_INFO raises exception, use default values
            pass

        return build_info

    def _get_installed_packages(self) -> list[tuple[str, str]]:
        """Get installed packages as (name, version) tuples using pipdeptree or fallback to importlib.metadata"""
        # Try pipdeptree first (cleanest approach)
        try:
            import pipdeptree  # type: ignore

            tree = pipdeptree.get_installed_distributions()
            return [(pkg.project_name.lower(), pkg.version) for pkg in tree]
        except ImportError:
            pass

        # Fallback to importlib.metadata
        try:
            from importlib.metadata import distributions

            return [(dist.metadata["Name"].lower(), dist.version) for dist in distributions()]
        except ImportError:
            return []

    def _get_dependencies(self) -> dict[str, str]:
        """Get versions of installed dependencies"""
        packages = self._get_installed_packages()
        return dict(packages)

    def _get_project_dependencies(self) -> list[str]:
        """Get project dependencies"""
        packages = self._get_installed_packages()
        return [name for name, version in packages]

    def _run_version_command(self, args: argparse.Namespace) -> None:
        """Execute the version command"""
        import json

        llama_stack_version = self._get_package_version("llama-stack")

        # Default behavior: just show version like llama stack --version
        if not any([args.build_info, args.dependencies, args.all]):
            if args.output == "json":
                print(json.dumps({"llama_stack_version": llama_stack_version}))
            else:
                print(llama_stack_version)
            return

        # Extended behavior: show additional information
        llama_stack_client_version = self._get_package_version("llama-stack-client")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        version_info: dict[str, Any] = {
            "llama_stack_version": llama_stack_version,
            "llama_stack_client_version": llama_stack_client_version,
            "python_version": python_version,
        }

        # Add build info if requested
        if args.build_info or args.all:
            build_info = self._get_build_info()
            version_info.update(
                {
                    "git_commit": build_info["commit_hash"],
                    "git_commit_date": build_info["commit_date"],
                    "git_branch": build_info["branch"],
                    "git_tag": build_info["tag"],
                    "build_timestamp": build_info["build_timestamp"],
                }
            )

        # Add dependencies if requested
        if args.dependencies or args.all:
            version_info["dependencies"] = self._get_dependencies()

        if args.output == "json":
            print(json.dumps(version_info))
        else:
            # Table format
            print("Llama Stack Version Information")
            print("=" * 50)

            # Build simple rows
            rows = [
                ["Llama Stack", version_info["llama_stack_version"]],
                ["Llama Stack Client", version_info["llama_stack_client_version"]],
                ["Python", version_info["python_version"]],
            ]

            # Add build info if requested
            if args.build_info or args.all:
                rows.extend(
                    [
                        ["", ""],  # separator
                        ["Git Commit", version_info["git_commit"]],
                        ["Commit Date", version_info["git_commit_date"]],
                        ["Git Branch", version_info["git_branch"]],
                        ["Git Tag", version_info["git_tag"]],
                        ["Build Timestamp", version_info["build_timestamp"]],
                    ]
                )

            # Add dependencies if requested
            if args.dependencies or args.all:
                deps = self._get_dependencies()
                rows.extend(
                    [
                        ["", ""],  # separator
                        ["Dependencies", f"{len(deps)} packages"],
                    ]
                )
                for dep, ver in sorted(deps.items()):
                    rows.append([f"  {dep}", ver])

            # Print simple table
            print_simple_table(rows)
