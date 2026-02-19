# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import OpenSearchVectorIOConfig


async def get_provider_impl(config: OpenSearchVectorIOConfig, deps: dict[Api, Any]):
    from llama_stack.providers.remote.vector_io.opensearch.opensearch import (
        OpenSearchVectorIOAdapter,
    )

    impl = OpenSearchVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files))
    await impl.initialize()
    return impl
