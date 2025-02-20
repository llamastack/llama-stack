# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
)

MODEL_ALIASES = [
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-Guard-3-8B",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_model_alias(
        "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]
