# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.log import setup_logging

from .stack import StackParser
from .stack.utils import print_subcommand_description


class LlamaCLIParser:
    """Defines CLI parser for Llama CLI"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="llama",
            description="Welcome to the Llama CLI",
            add_help=True,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Default command is to print help
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="subcommands")

        # Add sub-commands
        StackParser.create(subparsers)

        print_subcommand_description(self.parser, subparsers)

    def parse_args(self) -> argparse.Namespace:
        args = self.parser.parse_args()
        if not isinstance(args, argparse.Namespace):
            raise TypeError(f"Expected argparse.Namespace, got {type(args)}")
        return args

    def run(self, args: argparse.Namespace) -> None:
        args.func(args)


def main():
    # Initialize logging from environment variables before any other operations
    setup_logging()

    parser = LlamaCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
