# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace

from openai.types.completion_usage import CompletionUsage

from llama_stack.providers.utils.responses.responses_store import (
    _OpenAIResponseObjectWithInputAndMessages,
)
from llama_stack_api import (
    OpenAIAssistantMessageParam,
    OpenAIResponseMessage,
    OpenAIUserMessageParam,
)


async def test_compact_previous_response_uses_full_stored_messages_when_input_is_none(
    openai_responses_impl,
    mock_responses_store,
    mock_inference_api,
):
    previous_response = _OpenAIResponseObjectWithInputAndMessages(
        id="resp_prev",
        created_at=1,
        model="test-model",
        status="completed",
        store=True,
        # Simulate conversation-mode storage where only the latest turn is in input/output.
        input=[OpenAIResponseMessage(role="user", content="latest user turn", id="msg_latest_user")],
        output=[OpenAIResponseMessage(role="assistant", content="latest assistant turn", id="msg_latest_assistant")],
        # Full history is available in messages.
        messages=[
            OpenAIUserMessageParam(content="older user context"),
            OpenAIAssistantMessageParam(content="older assistant context"),
            OpenAIUserMessageParam(content="latest user turn"),
            OpenAIAssistantMessageParam(content="latest assistant turn"),
        ],
    )
    mock_responses_store.get_response_object.return_value = previous_response

    mock_inference_api.openai_chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="summary"))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    await openai_responses_impl.compact_openai_response(
        model="test-model",
        previous_response_id="resp_prev",
    )

    params = mock_inference_api.openai_chat_completion.await_args.args[0]
    user_messages = [msg.content for msg in params.messages if isinstance(msg, OpenAIUserMessageParam)]

    assert "older user context" in user_messages
