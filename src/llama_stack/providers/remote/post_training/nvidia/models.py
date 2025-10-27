# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
    build_hf_repo_model_entry,
)

_MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta/llama-3.1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
]


def get_model_entries() -> list[ProviderModelEntry]:
    return _MODEL_ENTRIES
