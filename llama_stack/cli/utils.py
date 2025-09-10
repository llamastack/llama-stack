# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.log import get_logger

logger = get_logger(name=__name__, category="cli")


# TODO: this can probably just be inlined now?
def add_config_distro_args(parser: argparse.ArgumentParser):
    """Add unified config/distro arguments."""
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "config",
        nargs="?",
        help="Configuration file path or distribution name",
    )


def get_config_from_args(args: argparse.Namespace) -> str | None:
    if args.config is not None:
        return str(args.config)
    return None
