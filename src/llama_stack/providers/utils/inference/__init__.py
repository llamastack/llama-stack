# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Static mapping from HuggingFace repo names to Llama model descriptors.
# This is the only data the server needs at runtime from the model registry.
#
# To regenerate after adding new models to sku_list.py, run:
#   uv run python3 -c "
#   from llama_stack.models.llama.sku_list import all_registered_models
#   d = {m.huggingface_repo: m.descriptor() for m in all_registered_models()}
#   for k, v in d.items():
#       print(f'    \"{k}\": \"{v}\",')
#   "
ALL_HUGGINGFACE_REPOS_TO_MODEL_DESCRIPTOR: dict[str, str] = {
    "meta-llama/Llama-2-7b": "Llama-2-7b",
    "meta-llama/Llama-2-13b": "Llama-2-13b",
    "meta-llama/Llama-2-70b": "Llama-2-70b",
    "meta-llama/Llama-2-7b-chat": "Llama-2-7b-chat",
    "meta-llama/Llama-2-13b-chat": "Llama-2-13b-chat",
    "meta-llama/Llama-2-70b-chat": "Llama-2-70b-chat",
    "meta-llama/Llama-3-8B": "Llama-3-8B",
    "meta-llama/Llama-3-70B": "Llama-3-70B",
    "meta-llama/Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "meta-llama/Llama-3-70B-Instruct": "Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-8B": "Llama3.1-8B",
    "meta-llama/Llama-3.1-70B": "Llama3.1-70B",
    "meta-llama/Llama-3.1-405B": "Llama3.1-405B:bf16-mp16",
    "meta-llama/Llama-3.1-405B-FP8": "Llama3.1-405B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama3.1-70B-Instruct",
    "meta-llama/Llama-3.1-405B-Instruct": "Llama3.1-405B-Instruct:bf16-mp16",
    "meta-llama/Llama-3.1-405B-Instruct-FP8": "Llama3.1-405B-Instruct",
    "meta-llama/Llama-3.2-1B": "Llama3.2-1B",
    "meta-llama/Llama-3.2-3B": "Llama3.2-3B",
    "meta-llama/Llama-3.2-11B-Vision": "Llama3.2-11B-Vision",
    "meta-llama/Llama-3.2-90B-Vision": "Llama3.2-90B-Vision",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8": "Llama3.2-1B-Instruct:int4-qlora-eo8",
    "meta-llama/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8": "Llama3.2-1B-Instruct:int4-spinquant-eo8",
    "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8": "Llama3.2-3B-Instruct:int4-qlora-eo8",
    "meta-llama/Llama-3.2-3B-Instruct-SpinQuant_INT4_EO8": "Llama3.2-3B-Instruct:int4-spinquant-eo8",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "Llama3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct": "Llama3.2-90B-Vision-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3-70B-Instruct",
    "meta-llama/Llama-4-Scout-17B-16E": "Llama-4-Scout-17B-16E",
    "meta-llama/Llama-4-Maverick-17B-128E": "Llama-4-Maverick-17B-128E",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "Llama-4-Maverick-17B-128E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama-4-Maverick-17B-128E-Instruct:fp8",
    "meta-llama/Llama-Guard-4-12B": "Llama-Guard-4-12B",
    "meta-llama/Llama-Guard-3-11B-Vision": "Llama-Guard-3-11B-Vision",
    "meta-llama/Llama-Guard-3-1B-INT4": "Llama-Guard-3-1B:int4",
    "meta-llama/Llama-Guard-3-1B": "Llama-Guard-3-1B",
    "meta-llama/Llama-Guard-3-8B": "Llama-Guard-3-8B",
    "meta-llama/Llama-Guard-3-8B-INT8": "Llama-Guard-3-8B:int8",
    "meta-llama/Llama-Guard-2-8B": "Llama-Guard-2-8B",
}
