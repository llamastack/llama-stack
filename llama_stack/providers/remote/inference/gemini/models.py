# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models import ModelType
from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

LLM_MODEL_IDS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

SAFETY_MODELS_ENTRIES = []

MODEL_ENTRIES = (
    [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS]
    + [
        ProviderModelEntry(
            provider_model_id="text-embedding-004",
            model_type=ModelType.embedding,
            metadata={"embedding_dimension": 768, "context_length": 2048},
        ),
    ]
    + SAFETY_MODELS_ENTRIES
)
