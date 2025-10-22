# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

import pytest

from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChoice,
    OpenAIUserMessageParam,
    Order,
)
from llama_stack.core.storage.datatypes import InferenceStoreReference, SqliteSqlStoreConfig
from llama_stack.providers.utils.inference.inference_store import InferenceStore
from llama_stack.providers.utils.sqlstore.sqlstore import register_sqlstore_backends


@pytest.fixture(autouse=True)
def setup_backends(tmp_path):
    """Register SQL store backends for testing."""
    db_path = str(tmp_path / "test.db")
    register_sqlstore_backends({"sql_default": SqliteSqlStoreConfig(db_path=db_path)})


def create_test_chat_completion(
    completion_id: str, created_timestamp: int, model: str = "test-model"
) -> OpenAIChatCompletion:
    """Helper to create a test chat completion."""
    return OpenAIChatCompletion(
        id=completion_id,
        created=created_timestamp,
        model=model,
        object="chat.completion",
        choices=[
            OpenAIChoice(
                index=0,
                message=OpenAIAssistantMessageParam(
                    role="assistant",
                    content=f"Response for {completion_id}",
                ),
                finish_reason="stop",
            )
        ],
    )


async def test_inference_store_pagination_basic():
    """Test basic pagination functionality."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data with different timestamps
    base_time = int(time.time())
    test_data = [
        ("zebra-task", base_time + 1),
        ("apple-job", base_time + 2),
        ("moon-work", base_time + 3),
        ("banana-run", base_time + 4),
        ("car-exec", base_time + 5),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test 1: First page with limit=2, descending order (default)
    result = await store.list_chat_completions(limit=2, order=Order.desc)
    assert len(result.data) == 2
    assert result.data[0].id == "car-exec"  # Most recent first
    assert result.data[1].id == "banana-run"
    assert result.has_more is True
    assert result.last_id == "banana-run"

    # Test 2: Second page using 'after' parameter
    result2 = await store.list_chat_completions(after="banana-run", limit=2, order=Order.desc)
    assert len(result2.data) == 2
    assert result2.data[0].id == "moon-work"
    assert result2.data[1].id == "apple-job"
    assert result2.has_more is True

    # Test 3: Final page
    result3 = await store.list_chat_completions(after="apple-job", limit=2, order=Order.desc)
    assert len(result3.data) == 1
    assert result3.data[0].id == "zebra-task"
    assert result3.has_more is False


async def test_inference_store_pagination_ascending():
    """Test pagination with ascending order."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data
    base_time = int(time.time())
    test_data = [
        ("delta-item", base_time + 1),
        ("charlie-task", base_time + 2),
        ("alpha-work", base_time + 3),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test ascending order pagination
    result = await store.list_chat_completions(limit=1, order=Order.asc)
    assert len(result.data) == 1
    assert result.data[0].id == "delta-item"  # Oldest first
    assert result.has_more is True

    # Second page with ascending order
    result2 = await store.list_chat_completions(after="delta-item", limit=1, order=Order.asc)
    assert len(result2.data) == 1
    assert result2.data[0].id == "charlie-task"
    assert result2.has_more is True


async def test_inference_store_pagination_with_model_filter():
    """Test pagination combined with model filtering."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data with different models
    base_time = int(time.time())
    test_data = [
        ("xyz-task", base_time + 1, "model-a"),
        ("def-work", base_time + 2, "model-b"),
        ("pqr-job", base_time + 3, "model-a"),
        ("abc-run", base_time + 4, "model-b"),
    ]

    # Store test chat completions
    for completion_id, timestamp, model in test_data:
        completion = create_test_chat_completion(completion_id, timestamp, model)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test pagination with model filter
    result = await store.list_chat_completions(limit=1, model="model-a", order=Order.desc)
    assert len(result.data) == 1
    assert result.data[0].id == "pqr-job"  # Most recent model-a
    assert result.data[0].model == "model-a"
    assert result.has_more is True

    # Second page with model filter
    result2 = await store.list_chat_completions(after="pqr-job", limit=1, model="model-a", order=Order.desc)
    assert len(result2.data) == 1
    assert result2.data[0].id == "xyz-task"
    assert result2.data[0].model == "model-a"
    assert result2.has_more is False


async def test_inference_store_pagination_invalid_after():
    """Test error handling for invalid 'after' parameter."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Try to paginate with non-existent ID
    with pytest.raises(ValueError, match="Record with id='non-existent' not found in table 'chat_completions'"):
        await store.list_chat_completions(after="non-existent", limit=2)


async def test_inference_store_pagination_no_limit():
    """Test pagination behavior when no limit is specified."""
    reference = InferenceStoreReference(backend="sql_default", table_name="chat_completions")
    store = InferenceStore(reference, policy=[])
    await store.initialize()

    # Create test data
    base_time = int(time.time())
    test_data = [
        ("omega-first", base_time + 1),
        ("beta-second", base_time + 2),
    ]

    # Store test chat completions
    for completion_id, timestamp in test_data:
        completion = create_test_chat_completion(completion_id, timestamp)
        input_messages = [OpenAIUserMessageParam(role="user", content=f"Test message for {completion_id}")]
        await store.store_chat_completion(completion, input_messages)

    # Wait for all queued writes to complete
    await store.flush()

    # Test without limit
    result = await store.list_chat_completions(order=Order.desc)
    assert len(result.data) == 2
    assert result.data[0].id == "beta-second"  # Most recent first
    assert result.data[1].id == "omega-first"
    assert result.has_more is False
