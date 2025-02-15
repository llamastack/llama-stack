# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
)

model_aliases = [
    build_model_alias(
        "llama3.1-8b",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "llama-3.3-70b",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
]
