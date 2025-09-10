# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import Api

from .config import HuggingFacePostTrainingConfig

# post_training api and the huggingface provider is still experimental and under heavy development


async def get_provider_impl(
    config: HuggingFacePostTrainingConfig,
    deps: dict[Api, Any],
):
    from .post_training import HuggingFacePostTrainingImpl

    impl = HuggingFacePostTrainingImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    return impl
