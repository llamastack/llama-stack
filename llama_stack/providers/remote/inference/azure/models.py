# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

# https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/models?tabs=global-standard%2Cstandard-chat-completions
LLM_MODEL_IDS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-chat",
    "o1",
    "o1-mini",
    "o3-mini",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

SAFETY_MODELS_ENTRIES = list[ProviderModelEntry]()

MODEL_ENTRIES = [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS] + SAFETY_MODELS_ENTRIES
