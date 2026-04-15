# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.providers.inline.inference.sentence_transformers.config import (
    SentenceTransformersInferenceConfig,
)


async def get_provider_impl(
    config: SentenceTransformersInferenceConfig,
    _deps: dict[str, Any],
):
    from .sentence_transformers import SentenceTransformersInferenceImpl

    impl = SentenceTransformersInferenceImpl(config)
    await impl.initialize()
    return impl
