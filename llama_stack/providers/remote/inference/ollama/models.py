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
    build_model_entry,
)

SAFETY_MODELS_ENTRIES = [
    # The Llama Guard models don't have their full fp16 versions
    # so we are going to alias their default version to the canonical SKU
    build_hf_repo_model_entry(
        "llama-guard3:8b",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_hf_repo_model_entry(
        "llama-guard3:1b",
        CoreModelId.llama_guard_3_1b.value,
    ),
]

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "llama3.1:8b-instruct-fp16",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_entry(
        "llama3.1:8b",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.1:70b-instruct-fp16",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_entry(
        "llama3.1:70b",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.1:405b-instruct-fp16",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_entry(
        "llama3.1:405b",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.2:1b-instruct-fp16",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_entry(
        "llama3.2:1b",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.2:3b-instruct-fp16",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_entry(
        "llama3.2:3b",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.2-vision:11b-instruct-fp16",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_entry(
        "llama3.2-vision:latest",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.2-vision:90b-instruct-fp16",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_entry(
        "llama3.2-vision:90b",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3.3:70b",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    ProviderModelEntry(
        provider_model_id="all-minilm:l6-v2",
        aliases=["all-minilm"],
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
            "context_length": 512,
        },
    ),
    ProviderModelEntry(
        provider_model_id="nomic-embed-text",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 768,
            "context_length": 8192,
        },
    ),
] + SAFETY_MODELS_ENTRIES
