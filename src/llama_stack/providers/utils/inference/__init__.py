# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.models.llama.sku_types import *  # noqa: F403


def is_supported_safety_model(model: Model) -> bool:
    if model.quantization_format != CheckpointQuantizationFormat.bf16:
        return False

    model_id = model.core_model_id
    return model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.llama_guard_3_1b,
        CoreModelId.llama_guard_3_11b_vision,
    ]


def supported_inference_models() -> list[Model]:
    return [
        m
        for m in all_registered_models()
        if (
            m.model_family in {ModelFamily.llama3_1, ModelFamily.llama3_2, ModelFamily.llama3_3, ModelFamily.llama4}
            or is_supported_safety_model(m)
        )
    ]


ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR = {m.huggingface_repo: m.descriptor() for m in all_registered_models()}
