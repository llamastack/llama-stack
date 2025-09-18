# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models import ModelType
from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
    build_hf_repo_model_entry,
)

SAFETY_MODELS_ENTRIES = [
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-guard-3-8b",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-guard-3-11b-vision",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama4-scout-instruct-basic",
        CoreModelId.llama4_scout_17b_16e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "accounts/fireworks/models/llama4-maverick-instruct-basic",
        CoreModelId.llama4_maverick_17b_128e_instruct.value,
    ),
    ProviderModelEntry(
        provider_model_id="nomic-ai/nomic-embed-text-v1.5",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    ),
] + SAFETY_MODELS_ENTRIES
