# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, IAny, nc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Callable

import torch
from pydantic import BaseModel
from torchtune.data._messages import InputOutputToMessages, ShareGPTToMessages
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.models.llama3_1 import lora_llama3_1_8b
from torchtune.models.llama3_2 import lora_llama3_2_3b
from torchtune.modules.transforms import Transform

from llama_stack.apis.post_training import DatasetFormat
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import Model

BuildLoraModelCallable = Callable[..., torch.nn.Module]
BuildTokenizerCallable = Callable[..., Llama3Tokenizer]


class ModelConfig(BaseModel):
    model_definition: BuildLoraModelCallable
    tokenizer_type: BuildTokenizerCallable
    checkpoint_type: str


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "Llama3.2-3B-Instruct": ModelConfig(
        model_definition=lora_llama3_2_3b,
        tokenizer_type=llama3_tokenizer,
        checkpoint_type="LLAMA3_2",
    ),
    "Llama3.1-8B-Instruct": ModelConfig(
        model_definition=lora_llama3_1_8b,
        tokenizer_type=llama3_tokenizer,
        checkpoint_type="LLAMA3",
    ),
}

DATA_FORMATS: dict[str, Transform] = {
    "instruct": InputOutputToMessages,
    "dialog": ShareGPTToMessages,
}


def _validate_model_id(model_id: str) -> Model:
    model = resolve_model(model_id)
    if model is None or model.core_model_id.value not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_id} is not supported.")
    return model


async def get_model_definition(
    model_id: str,
) -> BuildLoraModelCallable:
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "model_definition"):
        raise ValueError(f"Model {model_id} does not have model definition.")
    return model_config.model_definition


async def get_tokenizer_type(
    model_id: str,
) -> BuildTokenizerCallable:
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "tokenizer_type"):
        raise ValueError(f"Model {model_id} does not have tokenizer_type.")
    return model_config.tokenizer_type


async def get_checkpointer_model_type(
    model_id: str,
) -> str:
    """
    checkpointer model type is used in checkpointer for some special treatment on some specific model types
    For example, llama3.2 model tied weights (https://github.com/pytorch/torchtune/blob/main/torchtune/training/checkpointing/_checkpointer.py#L1041)
    """
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "checkpoint_type"):
        raise ValueError(f"Model {model_id} does not have checkpoint_type.")
    return model_config.checkpoint_type


async def get_data_transform(data_format: DatasetFormat) -> Transform:
    return DATA_FORMATS[data_format.value]
