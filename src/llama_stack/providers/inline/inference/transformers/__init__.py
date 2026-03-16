# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.inline.inference.transformers.config import (
    TransformersInferenceConfig,
)


async def get_provider_impl(
    config: TransformersInferenceConfig,
    _deps: dict[str, Any],
):
    from .transformers import TransformersInferenceImpl

    impl = TransformersInferenceImpl(config)
    await impl.initialize()
    return impl
