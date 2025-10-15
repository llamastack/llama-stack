# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.providers.datatypes import Api

from .config import ChromaVectorIOConfig


async def get_provider_impl(
    config: ChromaVectorIOConfig, deps: dict[Api, Any], run_config: StackRunConfig | None = None
):
    from llama_stack.providers.remote.vector_io.chroma.chroma import (
        ChromaVectorIOAdapter,
    )

    vector_stores_config = None
    if run_config and run_config.vector_stores:
        vector_stores_config = run_config.vector_stores

    impl = ChromaVectorIOAdapter(
        config,
        deps[Api.inference],
        deps[Api.models],
        deps.get(Api.files),
        vector_stores_config,
    )
    await impl.initialize()
    return impl
