# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import Any

from pydantic import BaseModel

from llama_stack.apis.post_training import TrainingConfig
from llama_stack.log import get_logger
from llama_stack.providers.remote.post_training.nvidia.config import SFTLoRADefaultConfig

from .config import NvidiaPostTrainingConfig

logger = get_logger(name=__name__, category="post_training::nvidia")


def warn_unsupported_params(config_dict: Any, supported_keys: set[str], config_name: str) -> None:
    keys = set(config_dict.__annotations__.keys()) if isinstance(config_dict, BaseModel) else config_dict.keys()
    unsupported_params = [k for k in keys if k not in supported_keys]
    if unsupported_params:
        warnings.warn(
            f"Parameters: {unsupported_params} in `{config_name}` not supported and will be ignored.", stacklevel=2
        )


def validate_training_params(
    training_config: dict[str, Any], supported_keys: set[str], config_name: str = "TrainingConfig"
) -> None:
    """
    Validates training parameters against supported keys.

    Args:
        training_config: Dictionary containing training configuration parameters
        supported_keys: Set of supported parameter keys
        config_name: Name of the configuration for warning messages
    """
    sft_lora_fields = set(SFTLoRADefaultConfig.__annotations__.keys())
    training_config_fields = set(TrainingConfig.__annotations__.keys())

    # Check for not supported parameters:
    # - not in either of configs
    # - in TrainingConfig but not in SFTLoRADefaultConfig
    unsupported_params = []
    for key in training_config:
        if isinstance(key, str) and key not in (supported_keys.union(sft_lora_fields)):
            if key in (not sft_lora_fields or training_config_fields):
                unsupported_params.append(key)

    if unsupported_params:
        warnings.warn(
            f"Parameters: {unsupported_params} in `{config_name}` are not supported and will be ignored.", stacklevel=2
        )


# ToDo: implement post health checks for customizer are enabled
async def _get_health(url: str) -> tuple[bool, bool]: ...


async def check_health(config: NvidiaPostTrainingConfig) -> None: ...
