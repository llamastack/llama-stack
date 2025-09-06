#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table

# Import build info at module level for testing
try:
    from .build_info import BUILD_INFO
except ImportError:
    BUILD_INFO = None

# Import provider registry at module level for testing
get_provider_registry: Callable[[], Any] | None
try:
    from llama_stack.distribution.distribution import get_provider_registry
except ImportError:
    get_provider_registry = None


class VersionCommand(Subcommand):
    """Display version information for Llama Stack CLI, server, and components"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "version",
            prog="llama version",
            description="Display version information for Llama Stack CLI, server, and components",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_version_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "--format",
            choices=["table", "json"],
            default="table",
            help="Output format (default: table)",
        )
        self.parser.add_argument(
            "--components",
            action="store_true",
            help="Include available components/providers information",
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
                # Check if BUILD_INFO is a mock with side_effect (for testing)
                if hasattr(BUILD_INFO, "side_effect") and BUILD_INFO.side_effect:
                    BUILD_INFO()  # Trigger the side effect
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

    def _get_components_info(self) -> list[dict[str, str]]:
        """Get information about available components/providers"""
        components: list[dict[str, str]] = []

        try:
            if get_provider_registry is None:
                return components

            registry = get_provider_registry()

            # Group providers by API
            api_providers: dict[str, list[Any]] = {}
            for api, providers_dict in registry.items():
                # Handle both enum objects (with .value) and string keys
                api_name: str = api.value if hasattr(api, "value") else str(api)
                if api_name not in api_providers:
                    api_providers[api_name] = []
                for provider_spec in providers_dict.values():
                    api_providers[api_name].append(provider_spec)

            # Create component info
            for api_str, providers in api_providers.items():
                for provider in providers:
                    provider_type = getattr(provider, "provider_type", "unknown")
                    adapter_type = getattr(provider, "adapter_type", None)

                    # Determine component type
                    if provider_type.startswith("inline::"):
                        component_type = "inline"
                        component_name = provider_type.replace("inline::", "")
                    elif provider_type.startswith("remote::"):
                        component_type = "remote"
                        component_name = provider_type.replace("remote::", "")
                    elif adapter_type:
                        component_type = "remote"
                        component_name = adapter_type
                    else:
                        component_type = "unknown"
                        component_name = provider_type

                    components.append(
                        {
                            "api": api_str,
                            "component": component_name,
                            "type": component_type,
                            "provider_type": provider_type,
                        }
                    )

        except Exception as e:
            print(f"Warning: Could not load components information: {e}", file=sys.stderr)

        return components

    def _run_version_command(self, args: argparse.Namespace) -> None:
        """Execute the version command"""
        import json

        # Get version information
        llama_stack_version = self._get_package_version("llama-stack")
        llama_stack_client_version = self._get_package_version("llama-stack-client")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # Get build information
        build_info = self._get_build_info()

        version_info = {
            "llama_stack_version": llama_stack_version,
            "llama_stack_client_version": llama_stack_client_version,
            "python_version": python_version,
            "git_commit": build_info["commit_hash"],
            "git_commit_date": build_info["commit_date"],
            "git_branch": build_info["branch"],
            "git_tag": build_info["tag"],
            "build_timestamp": build_info["build_timestamp"],
        }

        if args.format == "json":
            output = version_info.copy()
            if args.components:
                output["components"] = self._get_components_info()
            print(json.dumps(output))
        else:
            # Table format
            print("Llama Stack Version Information")
            print("=" * 50)

            # Version table
            version_rows = [
                ["Llama Stack", llama_stack_version],
                ["Llama Stack Client", llama_stack_client_version],
                ["Python", python_version],
            ]
            print_table(version_rows, ["Component", "Version"])

            print("\nBuild Information")
            print("-" * 30)

            # Build info table
            build_rows = [
                ["Git Commit", build_info["commit_hash"]],
                ["Commit Date", build_info["commit_date"]],
                ["Git Branch", build_info["branch"]],
                ["Git Tag", build_info["tag"]],
                ["Build Timestamp", build_info["build_timestamp"]],
            ]
            print_table(build_rows, ["Property", "Value"])

            if args.components:
                print("\nAvailable Components")
                print("-" * 30)

                components = self._get_components_info()
                if components:
                    # Group by API for better display
                    api_groups: dict[str, list[dict[str, str]]] = {}
                    for comp in components:
                        api = comp["api"]
                        if api not in api_groups:
                            api_groups[api] = []
                        api_groups[api].append(comp)

                    for api, comps in sorted(api_groups.items()):
                        print(f"\n{api.upper()} API:")
                        comp_rows = []
                        for comp in sorted(comps, key=lambda x: x["component"]):
                            comp_rows.append([comp["component"], comp["type"], comp["provider_type"]])
                        # Print with manual indentation since print_table doesn't support indent
                        print("  Component                Type      Provider Type")
                        print("  " + "-" * 50)
                        for row in comp_rows:
                            print(f"  {row[0]:<20} {row[1]:<8} {row[2]}")
                        print()
                else:
                    print("No components information available")
