# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from .config_dirs import DEFAULT_CHECKPOINT_DIR


def model_local_dir(descriptor: str) -> str:
    return str(Path(DEFAULT_CHECKPOINT_DIR) / (descriptor.replace(":", "-")))
