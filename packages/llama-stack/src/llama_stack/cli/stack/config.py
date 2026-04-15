# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import sys
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand
from llama_stack.core.utils.config_resolution import resolve_config_or_distro, resolve_sole_distribution_package


class StackConfig(Subcommand):
    """CLI subcommand group for config operations."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "config",
            prog="llama stack config",
            description="Inspect and manage Llama Stack configurations.",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        config_subparsers = self.parser.add_subparsers(title="config_subcommands")
        StackConfigGenerate.create(config_subparsers)
        StackConfigShow.create(config_subparsers)


class StackConfigGenerate(Subcommand):
    """CLI subcommand to generate a run configuration from installed providers."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "generate",
            prog="llama stack config generate",
            description="Generate a run configuration from installed providers and optional patches.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_config_generate_command)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output file path. If not specified, writes to stdout.",
        )
        self.parser.add_argument(
            "--patch",
            type=str,
            default=None,
            help="Path to a patch YAML file to apply on top of the generated base config.",
        )

    def _run_config_generate_command(self, args: argparse.Namespace) -> None:
        from ._generate_config import run_generate_config_command

        run_generate_config_command(args)


class StackConfigShow(Subcommand):
    """CLI subcommand to display a resolved configuration."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "show",
            prog="llama stack config show",
            description="Print a resolved configuration to stdout.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_config_show_command)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            nargs="?",
            metavar="config | distro[::variant]",
            help="Config file path, distribution name, or distro::variant. "
            "If omitted, uses the sole installed distribution package.",
        )

    def _run_config_show_command(self, args: argparse.Namespace) -> None:
        config_path: Path | None = None
        if args.config:
            try:
                config_path = resolve_config_or_distro(args.config)
            except ValueError as e:
                print(str(e), file=sys.stderr)
                sys.exit(1)
        else:
            config_path = resolve_sole_distribution_package()
            if not config_path:
                print("No distribution package installed", file=sys.stderr)
                sys.exit(1)

        if config_path:
            sys.stdout.write(config_path.read_text())
