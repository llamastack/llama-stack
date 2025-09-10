# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shutil

from llama_stack.cli.subcommand import Subcommand
from llama_stack.core.utils.config_dirs import DEFAULT_CHECKPOINT_DIR
from llama_stack.models.llama.sku_list import resolve_model


class ModelRemove(Subcommand):
    """Remove the downloaded llama model"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "remove",
            prog="llama model remove",
            description="Remove the downloaded llama model",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_remove_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model",
            required=True,
            help="Specify the llama downloaded model name, see `llama model list --downloaded`",
        )
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Used to forcefully remove the llama model from the storage without further confirmation",
        )

    def _run_model_remove_cmd(self, args: argparse.Namespace) -> None:
        from .safety_models import prompt_guard_model_sku_map

        prompt_guard_model_map = prompt_guard_model_sku_map()

        if args.model in prompt_guard_model_map.keys():
            model = prompt_guard_model_map[args.model]
        else:
            model = resolve_model(args.model)

        model_path = os.path.join(DEFAULT_CHECKPOINT_DIR, args.model.replace(":", "-"))

        if model is None or not os.path.isdir(model_path):
            print(f"'{args.model}' is not a valid llama model or does not exist.")
            return

        if args.force:
            shutil.rmtree(model_path)
            print(f"{args.model} removed.")
        else:
            if input(f"Are you sure you want to remove {args.model}? (y/n): ").strip().lower() == "y":
                shutil.rmtree(model_path)
                print(f"{args.model} removed.")
            else:
                print("Removal aborted.")
