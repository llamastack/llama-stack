# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Regression tests for None-safe response content joins."""

from llama_stack.providers.inline.agents.builtin.responses.types import ChatCompletionResult


def _build_result(content: list[str | None]) -> ChatCompletionResult:
    return ChatCompletionResult(
        response_id="resp_123",
        content=content,
        tool_calls={},
        created=0,
        model="test-model",
        finish_reason="stop",
        message_item_id="msg_123",
        tool_call_item_ids={},
        content_part_emitted=False,
    )


def test_content_text_skips_none_entries():
    assert _build_result([None, "tool result"]).content_text == "tool result"


def test_content_text_returns_empty_string_for_none_only_content():
    assert _build_result([None]).content_text == ""
