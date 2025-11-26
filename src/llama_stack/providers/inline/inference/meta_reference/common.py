# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.core.utils.model_utils import model_local_dir


def model_checkpoint_dir(model_id) -> str:
    checkpoint_dir = Path(model_local_dir(model_id))

    paths = [Path(checkpoint_dir / f"consolidated.{ext}") for ext in ["pth", "00.pth"]]
    if not any(p.exists() for p in paths):
        checkpoint_dir = checkpoint_dir / "original"

    assert checkpoint_dir.exists(), (
        f"Could not find checkpoints in: {model_local_dir(model_id)}. "
        f"If you try to use the native llama model, please download the model using `llama-model download --source meta --model-id {model_id}` (see https://github.com/meta-llama/llama-models). "
        f"Otherwise, please save your model checkpoint under {model_local_dir(model_id)}"
    )
    return str(checkpoint_dir)
