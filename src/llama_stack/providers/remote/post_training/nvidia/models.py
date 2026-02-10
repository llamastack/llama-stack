# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.utils.inference.model_registry import build_hf_repo_model_entry

_MODEL_ENTRIES = [
    build_hf_repo_model_entry(
        "meta/llama-3.1-8b-instruct",
        "Llama3.1-8B-Instruct",
    ),
    build_hf_repo_model_entry(
        "meta/llama-3.2-1b-instruct",
        "Llama3.2-1B-Instruct",
    ),
]
