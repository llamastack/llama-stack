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
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
]

SAFETY_MODELS_ENTRIES = []

MODEL_ENTRIES = (
    [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS]
    + [
        ProviderModelEntry(
            provider_model_id="voyage-3",
            model_type=ModelType.embedding,
            metadata={"embedding_dimension": 1024, "context_length": 32000},
        ),
        ProviderModelEntry(
            provider_model_id="voyage-3-lite",
            model_type=ModelType.embedding,
            metadata={"embedding_dimension": 512, "context_length": 32000},
        ),
        ProviderModelEntry(
            provider_model_id="voyage-code-3",
            model_type=ModelType.embedding,
            metadata={"embedding_dimension": 1024, "context_length": 32000},
        ),
    ]
    + SAFETY_MODELS_ENTRIES
)
