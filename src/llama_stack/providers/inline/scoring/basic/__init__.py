# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.core.datatypes import Api

from .config import BasicScoringConfig


async def get_provider_impl(
    config: BasicScoringConfig,
    deps: dict[Api, Any],
):
    from .scoring import BasicScoringImpl

    impl = BasicScoringImpl(
        config,
        deps[Api.datasetio],
        deps[Api.datasets],
    )
    await impl.initialize()
    return impl
