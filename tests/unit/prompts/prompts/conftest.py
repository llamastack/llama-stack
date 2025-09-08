# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random

import pytest

from llama_stack.core.prompts.prompts import PromptServiceConfig, PromptServiceImpl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
async def temp_prompt_store(tmp_path_factory):
    unique_id = f"prompt_store_{random.randint(1, 1000000)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")

    config = PromptServiceConfig(kvstore=SqliteKVStoreConfig(db_path=db_path))
    store = PromptServiceImpl(config, deps={})
    await store.initialize()
    yield store
