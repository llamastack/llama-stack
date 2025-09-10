# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import time
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table
from llama_stack.core.utils.config_dirs import DEFAULT_CHECKPOINT_DIR
from llama_stack.models.llama.sku_list import all_registered_models


def _get_model_size(model_dir):
    return sum(f.stat().st_size for f in Path(model_dir).rglob("*") if f.is_file())


def _convert_to_model_descriptor(model):
    for m in all_registered_models():
        if model == m.descriptor().replace(":", "-"):
            return str(m.descriptor())
    return str(model)


def _run_model_list_downloaded_cmd() -> None:
    headers = ["Model", "Size", "Modified Time"]

    rows = []
    for model in os.listdir(DEFAULT_CHECKPOINT_DIR):
        abs_path = os.path.join(DEFAULT_CHECKPOINT_DIR, model)
        space_usage = _get_model_size(abs_path)
        model_size = f"{space_usage / (1024**3):.2f} GB"
        modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(abs_path)))
        rows.append(
            [
                _convert_to_model_descriptor(model),
                model_size,
                modified_time,
            ]
        )

    print_table(
        rows,
        headers,
        separate_rows=True,
    )


class ModelList(Subcommand):
    """List available llama models"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama model list",
            description="Show available llama models",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_list_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--show-all",
            action="store_true",
            help="Show all models (not just defaults)",
        )
        self.parser.add_argument(
            "--downloaded",
            action="store_true",
            help="List the downloaded models",
        )
        self.parser.add_argument(
            "-s",
            "--search",
            type=str,
            required=False,
            help="Search for the input string as a substring in the model descriptor(ID)",
        )

    def _run_model_list_cmd(self, args: argparse.Namespace) -> None:
        from .safety_models import prompt_guard_model_skus

        if args.downloaded:
            return _run_model_list_downloaded_cmd()

        headers = [
            "Model Descriptor(ID)",
            "Hugging Face Repo",
            "Context Length",
        ]

        rows = []
        for model in all_registered_models() + prompt_guard_model_skus():
            if not args.show_all and not model.is_featured:
                continue

            descriptor = model.descriptor()
            if not args.search or args.search.lower() in descriptor.lower():
                rows.append(
                    [
                        descriptor,
                        model.huggingface_repo,
                        f"{model.max_seq_length // 1024}K",
                    ]
                )
        if len(rows) == 0:
            print(f"Did not find any model matching `{args.search}`.")
        else:
            print_table(
                rows,
                headers,
                separate_rows=True,
            )
