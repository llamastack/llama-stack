# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Opt-in concurrency stress tests for the Responses↔Conversations sync path.

By default this module is skipped to avoid lengthening CI. Set the environment
variable LLAMA_STACK_ENABLE_CONCURRENCY_TESTS=1 to run it locally. Optional
knobs:

    LLAMA_STACK_CONCURRENCY_WORKERS   (default: 4)
    LLAMA_STACK_CONCURRENCY_TURNS     (default: 3)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from llama_stack.core.testing_context import reset_test_context, set_test_context

if not os.getenv("LLAMA_STACK_ENABLE_CONCURRENCY_TESTS"):
    pytest.skip("Set LLAMA_STACK_ENABLE_CONCURRENCY_TESTS=1 to run stress tests", allow_module_level=True)


CONCURRENCY_WORKERS = int(os.getenv("LLAMA_STACK_CONCURRENCY_WORKERS", "4"))
TURNS_PER_WORKER = int(os.getenv("LLAMA_STACK_CONCURRENCY_TURNS", "1"))

# Reuse the exact prompts from test_conversation_responses to hit existing recordings.
FIRST_TURN_PROMPT = "Say hello"
STREAMING_PROMPT = "Say goodbye"
RECORDED_TEST_NODE_ID = (
    "tests/integration/responses/test_conversation_responses.py::TestConversationResponses::"
    "test_conversation_multi_turn_and_streaming[txt=openai/gpt-4o]"
)


@pytest.mark.integration
def test_conversation_streaming_multi_client(openai_client, text_model_id):
    """Run many copies of the conversation streaming flow concurrently (matching recorded prompts)."""

    if CONCURRENCY_WORKERS < 2:
        pytest.skip("Need at least 2 workers for concurrency stress")

    def stream_single_turn(conversation_id: str, worker_idx: int, turn_idx: int):
        """Send the streaming turn and ensure completion."""
        response_stream = openai_client.responses.create(
            model=text_model_id,
            input=[{"role": "user", "content": STREAMING_PROMPT}],
            conversation=conversation_id,
            stream=True,
        )

        final_response = None
        failed_error: str | None = None
        for chunk in response_stream:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type is None and isinstance(chunk, dict):
                chunk_type = chunk.get("type")

            chunk_response = getattr(chunk, "response", None)
            if chunk_response is None and isinstance(chunk, dict):
                chunk_response = chunk.get("response")

            if chunk_type in {"response.completed", "response.incomplete"}:
                final_response = chunk_response
                break
            if chunk_type == "response.failed":
                error_obj = None
                if chunk_response is not None:
                    error_obj = getattr(chunk_response, "error", None)
                    if error_obj is None and isinstance(chunk_response, dict):
                        error_obj = chunk_response.get("error")
                if isinstance(error_obj, dict):
                    failed_error = error_obj.get("message", "response.failed")
                elif hasattr(error_obj, "message"):
                    failed_error = error_obj.message  # type: ignore[assignment]
                else:
                    failed_error = "response.failed event without error"
                break

        if failed_error:
            raise RuntimeError(f"Worker {worker_idx} turn {turn_idx} failed: {failed_error}")
        if final_response is None:
            raise RuntimeError(f"Worker {worker_idx} turn {turn_idx} did not complete")

    def worker(worker_idx: int) -> list[str]:
        token = set_test_context(RECORDED_TEST_NODE_ID)
        conversation_ids: list[str] = []
        try:
            for turn_idx in range(TURNS_PER_WORKER):
                conversation = openai_client.conversations.create()
                assert conversation.id.startswith("conv_")
                conversation_ids.append(conversation.id)

                openai_client.responses.create(
                    model=text_model_id,
                    input=[{"role": "user", "content": FIRST_TURN_PROMPT}],
                    conversation=conversation.id,
                )

                stream_single_turn(conversation.id, worker_idx, turn_idx)

            return conversation_ids
        finally:
            reset_test_context(token)

    all_conversation_ids: list[str] = []
    futures = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY_WORKERS) as executor:
        for worker_idx in range(CONCURRENCY_WORKERS):
            futures.append(executor.submit(worker, worker_idx))

        for future in as_completed(futures):
            all_conversation_ids.extend(future.result())

    expected_conversations = CONCURRENCY_WORKERS * TURNS_PER_WORKER
    assert len(all_conversation_ids) == expected_conversations

    for conversation_id in all_conversation_ids:
        conversation_items = openai_client.conversations.items.list(conversation_id)
        assert len(conversation_items.data) >= 4, (
            f"Conversation {conversation_id} missing expected items: {len(conversation_items.data)}"
        )
