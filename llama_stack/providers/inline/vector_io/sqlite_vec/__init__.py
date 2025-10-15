# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import StackRunConfig
from llama_stack.providers.datatypes import Api

from .config import SQLiteVectorIOConfig


async def get_provider_impl(
    config: SQLiteVectorIOConfig, deps: dict[Api, Any], run_config: StackRunConfig | None = None
):
    from .sqlite_vec import SQLiteVecVectorIOAdapter

    vector_stores_config = None
    if run_config and run_config.vector_stores:
        vector_stores_config = run_config.vector_stores

    assert isinstance(config, SQLiteVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = SQLiteVecVectorIOAdapter(
        config,
        deps[Api.inference],
        deps[Api.models],
        deps.get(Api.files),
        vector_stores_config,
    )
    await impl.initialize()
    return impl
