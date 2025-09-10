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
        "meta-llama/Llama-Guard-3-8B",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]
MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    ProviderModelEntry(
        provider_model_id="togethercomputer/m2-bert-80M-8k-retrieval",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    ),
    ProviderModelEntry(
        provider_model_id="togethercomputer/m2-bert-80M-32k-retrieval",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 32768,
        },
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        CoreModelId.llama4_scout_17b_16e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        CoreModelId.llama4_maverick_17b_128e_instruct.value,
    ),
] + SAFETY_MODELS_ENTRIES
