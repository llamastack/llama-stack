# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from pathlib import Path

LLAMA_STACK_CLIENT_CONFIG_DIR = Path(os.path.expanduser("~/.llama/client"))


def get_config_file_path():
    return LLAMA_STACK_CLIENT_CONFIG_DIR / "config.yaml"
