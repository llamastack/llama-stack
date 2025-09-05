# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import tempfile

import pytest

from llama_stack.core.prompts.prompts import PromptServiceConfig, PromptServiceImpl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
async def temp_prompt_store():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
        db_path = tmp_file.name

    try:
        config = PromptServiceConfig(kvstore=SqliteKVStoreConfig(db_path=db_path))
        store = PromptServiceImpl(config, deps={})
        await store.initialize()
        yield store
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.fixture
def sample_prompt_data():
    return {
        "prompt": "Hello {{name}}, welcome to {{platform}}!",
        "variables": {"name": "John", "platform": "LlamaStack"},
    }


@pytest.fixture
def sample_prompts_data():
    return [
        {"prompt": "Hello {{name}}!", "variables": {"name": "Alice"}},
        {"prompt": "Welcome to {{platform}}, {{user}}!", "variables": {"platform": "LlamaStack", "user": "Bob"}},
        {"prompt": "Your order {{order_id}} is ready for pickup.", "variables": {"order_id": "12345"}},
    ]
