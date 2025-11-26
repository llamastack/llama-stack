# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import Api

from .config import TorchtunePostTrainingConfig

# post_training api and the torchtune provider is still experimental and under heavy development


async def get_provider_impl(
    config: TorchtunePostTrainingConfig,
    deps: dict[Api, Any],
):
    from .post_training import TorchtunePostTrainingImpl

    impl = TorchtunePostTrainingImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    return impl
