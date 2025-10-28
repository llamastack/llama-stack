# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_hf_repo_model_entry,
)

SAFETY_MODELS_ENTRIES = []


# https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta.llama3-1-8b-instruct-v1:0",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta.llama3-1-70b-instruct-v1:0",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta.llama3-1-405b-instruct-v1:0",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
] + SAFETY_MODELS_ENTRIES
