# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.utils.inference.model_registry import (
    ProviderModelEntry,
)

# Vertex AI model IDs with vertex_ai/ prefix as required by litellm
LLM_MODEL_IDS = [
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.5-flash",
    "vertex_ai/gemini-2.5-pro",
]

SAFETY_MODELS_ENTRIES = list[ProviderModelEntry]()

MODEL_ENTRIES = [ProviderModelEntry(provider_model_id=m) for m in LLM_MODEL_IDS] + SAFETY_MODELS_ENTRIES
