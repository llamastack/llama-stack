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

SAFETY_MODELS_ENTRIES = []

# https://docs.nvidia.com/nim/large-language-models/latest/supported-llm-agnostic-architectures.html
MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta/llama3-8b-instruct",
        CoreModelId.llama3_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama3-70b-instruct",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.3-70b-instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    ProviderModelEntry(
        provider_model_id="nvidia/vila",
        model_type=ModelType.llm,
    ),
    # NeMo Retriever Text Embedding models -
    #
    # https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/support-matrix.html
    #
    # +-----------------------------------+--------+-----------+-----------+------------+
    # | Model ID                          | Max    | Publisher | Embedding | Dynamic    |
    # |                                   | Tokens |           | Dimension | Embeddings |
    # +-----------------------------------+--------+-----------+-----------+------------+
    # | nvidia/llama-3.2-nv-embedqa-1b-v2 | 8192   | NVIDIA    | 2048      | Yes        |
    # | nvidia/nv-embedqa-e5-v5           |  512   | NVIDIA    | 1024      |  No        |
    # | nvidia/nv-embedqa-mistral-7b-v2   |  512   | NVIDIA    | 4096      |  No        |
    # | snowflake/arctic-embed-l          |  512   | Snowflake | 1024      |  No        |
    # +-----------------------------------+--------+-----------+-----------+------------+
    ProviderModelEntry(
        provider_model_id="nvidia/llama-3.2-nv-embedqa-1b-v2",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 2048,
            "context_length": 8192,
        },
    ),
    ProviderModelEntry(
        provider_model_id="nvidia/nv-embedqa-e5-v5",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 1024,
            "context_length": 512,
        },
    ),
    ProviderModelEntry(
        provider_model_id="nvidia/nv-embedqa-mistral-7b-v2",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 4096,
            "context_length": 512,
        },
    ),
    ProviderModelEntry(
        provider_model_id="snowflake/arctic-embed-l",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 1024,
            "context_length": 512,
        },
    ),
    # TODO(mf): how do we handle Nemotron models?
    # "Llama3.1-Nemotron-51B-Instruct" -> "meta/llama-3.1-nemotron-51b-instruct",
] + SAFETY_MODELS_ENTRIES
