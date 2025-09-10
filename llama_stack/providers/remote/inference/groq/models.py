# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.sku_list import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_hf_repo_model_entry,
    build_model_entry,
)

SAFETY_MODELS_ENTRIES = []

MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "llama3-8b-8192",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_entry(
        "llama-3.1-8b-instant",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama3-70b-8192",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "llama-3.3-70b-versatile",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    # Groq only contains a preview version for llama-3.2-3b
    # Preview models aren't recommended for production use, but we include this one
    # to pass the test fixture
    # TODO(aidand): Replace this with a stable model once Groq supports it
    build_hf_repo_model_entry(
        "llama-3.2-3b-preview",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-4-scout-17b-16e-instruct",
        CoreModelId.llama4_scout_17b_16e_instruct.value,
    ),
    build_hf_repo_model_entry(
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        CoreModelId.llama4_maverick_17b_128e_instruct.value,
    ),
] + SAFETY_MODELS_ENTRIES
