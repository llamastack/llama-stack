# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
from pathlib import Path

from llama_stack.log import get_logger

logger = get_logger(__name__, "models")


def load_bpe_file(model_path: Path) -> dict[bytes, int]:
    """
    Load BPE file directly and return mergeable ranks.

    Args:
        model_path (Path): Path to the BPE model file.

    Returns:
        dict[bytes, int]: Dictionary mapping byte sequences to their ranks.
    """
    mergeable_ranks = {}

    with open(model_path, encoding="utf-8") as f:
        content = f.read()

    for line in content.splitlines():
        if not line.strip():  # Skip empty lines
            continue
        try:
            token, rank = line.split()
            mergeable_ranks[base64.b64decode(token)] = int(rank)
        except Exception as e:
            logger.warning(f"Failed to parse line '{line}': {e}")
            continue

    return mergeable_ranks
