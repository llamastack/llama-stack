# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import argparse
import textwrap

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.subcommand import Subcommand


class StackShow(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "show",
            prog="llama stack show",
            description="show the dependencies for a llama stack distribution",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_show_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distributions/**/build.yaml. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--distro",
            type=str,
            default=None,
            help="Name of the distro config to use for show. You may use `llama stack show --list-distros` to check out the available distros",
        )

        self.parser.add_argument(
            "--list-distros",
            action="store_true",
            default=False,
            help="Show the available templates for building a Llama Stack distribution",
        )

        self.parser.add_argument(
            "--env-name",
            type=str,
            help=textwrap.dedent(
                f"""[for image-type={"|".join(e.value for e in ImageType)}] Name of the conda or virtual environment to use for
the build. If not specified, currently active environment will be used if found.
            """
            ),
            default=None,
        )
        self.parser.add_argument(
            "--providers",
            type=str,
            default=None,
            help="sync dependencies for a list of providers and only those providers. This list is formatted like: api1=provider1,api2=provider2. Where there can be multiple providers per API.",
        )

    def _run_stack_show_command(self, args: argparse.Namespace) -> None:
        # always keep implementation completely silo-ed away from CLI so CLI
        # can be fast to load and reduces dependencies
        from ._show import run_stack_show_command

        return run_stack_show_command(args)
