# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import argparse
import textwrap

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.subcommand import Subcommand


class StackBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama stack build",
            description="Build a Llama stack container",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_build_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to a config file to use for the build. You can find example configs in llama_stack.cores/**/build.yaml. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--template",
            type=str,
            default=None,
            help="""(deprecated) Name of the example template config to use for build. You may use `llama stack build --list-distros` to check out the available distributions""",
        )
        self.parser.add_argument(
            "--distro",
            "--distribution",
            dest="distribution",
            type=str,
            default=None,
            help="""Name of the distribution to use for build. You may use `llama stack build --list-distros` to check out the available distributions""",
        )

        self.parser.add_argument(
            "--list-distros",
            "--list-distributions",
            action="store_true",
            dest="list_distros",
            default=False,
            help="Show the available distributions for building a Llama Stack distribution",
        )

        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type to use for the build. If not specified, will use the image type from the template config.",
            choices=[e.value for e in ImageType],
            default=None,  # no default so we can detect if a user specified --image-type and override image_type in the config
        )

        self.parser.add_argument(
            "--image-name",
            type=str,
            help=textwrap.dedent(
                f"""[for image-type={"|".join(e.value for e in ImageType)}] Name of the virtual environment to use for
the build. If not specified, currently active environment will be used if found.
            """
            ),
            default=None,
        )
        self.parser.add_argument(
            "--print-deps-only",
            default=False,
            action="store_true",
            help="Print the dependencies for the stack only, without building the stack",
        )

        self.parser.add_argument(
            "--run",
            action="store_true",
            default=False,
            help="Run the stack after building using the same image type, name, and other applicable arguments",
        )
        self.parser.add_argument(
            "--providers",
            type=str,
            default=None,
            help="Build a config for a list of providers and only those providers. This list is formatted like: api1=provider1,api2=provider2. Where there can be multiple providers per API.",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        # always keep implementation completely silo-ed away from CLI so CLI
        # can be fast to load and reduces dependencies
        from ._build import run_stack_build_command

        return run_stack_build_command(args)
