# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import MLflowPromptsConfig
from .mlflow import MLflowPromptsAdapter

__all__ = ["MLflowPromptsConfig", "MLflowPromptsAdapter", "get_adapter_impl"]


async def get_adapter_impl(config: MLflowPromptsConfig, _deps):
    """Get the MLflow prompts adapter implementation."""
    impl = MLflowPromptsAdapter(config=config)
    await impl.initialize()
    return impl
