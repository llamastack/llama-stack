# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand


class StackListApis(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-apis",
            prog="llama stack list-apis",
            description="List APIs part of the Llama Stack implementation",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_apis_list_cmd)

    def _add_arguments(self):
        pass

    def _run_apis_list_cmd(self, args: argparse.Namespace) -> None:
        from llama_stack.cli.table import print_table
        from llama_stack.core.distribution import stack_apis

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "API",
        ]

        rows = []
        for api in stack_apis():
            rows.append(
                [
                    api.value,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
