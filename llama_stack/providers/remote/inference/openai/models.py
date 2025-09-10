# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from dataclasses import dataclass

from llama_stack.apis.models import ModelType
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

LLM_MODEL_IDS = [
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "gpt-4o-audio-preview",
    "chatgpt-4o-latest",
    "o1",
    "o1-mini",
    "o3-mini",
    "o4-mini",
]


@dataclass
class EmbeddingModelInfo:
    """Structured representation of embedding model information."""

    embedding_dimension: int
    context_length: int


EMBEDDING_MODEL_IDS: dict[str, EmbeddingModelInfo] = {
    "text-embedding-3-small": EmbeddingModelInfo(1536, 8192),
    "text-embedding-3-large": EmbeddingModelInfo(3072, 8192),
}
SAFETY_MODELS_ENTRIES = []

MODEL_ENTRIES = (
    [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS]
    + [
        ProviderModelEntry(
            provider_model_id=model_id,
            model_type=ModelType.embedding,
            metadata={
                "embedding_dimension": model_info.embedding_dimension,
                "context_length": model_info.context_length,
            },
        )
        for model_id, model_info in EMBEDDING_MODEL_IDS.items()
    ]
    + SAFETY_MODELS_ENTRIES
)
