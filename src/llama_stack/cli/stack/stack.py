# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from importlib.metadata import version

from llama_stack.cli.stack.list_stacks import StackListBuilds
from llama_stack.cli.stack.utils import print_subcommand_description
from llama_stack.cli.subcommand import Subcommand

from .list_apis import StackListApis
from .list_deps import StackListDeps
from .list_providers import StackListProviders
from .remove import StackRemove
from .run import StackRun


class StackParser(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "stack",
            prog="llama stack",
            description="Operations for the Llama Stack / Distributions",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        self.parser.add_argument(
            "--version",
            action="version",
            version=f"{version('llama-stack')}",
        )

        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="stack_subcommands")

        # Add sub-commands
        StackListDeps.create(subparsers)
        StackListApis.create(subparsers)
        StackListProviders.create(subparsers)
        StackRun.create(subparsers)
        StackRemove.create(subparsers)
        StackListBuilds.create(subparsers)
        print_subcommand_description(self.parser, subparsers)
