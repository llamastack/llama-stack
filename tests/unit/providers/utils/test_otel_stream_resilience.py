# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Integration test: opentelemetry-instrumentation-openai-v2 crashes on tool-call
streaming when a provider sends ``arguments=None``.

This test imports the *real* OTEL wrapper classes and feeds them *real* OpenAI
SDK types.  Nothing is mocked or simulated — if the test passes, the crash is
confirmed in the actual library code.

Skipped automatically when opentelemetry-instrumentation-openai-v2 is not
installed.
"""

import pytest

# Skip the entire module if the OTEL instrumentation package is not installed.
pytest.importorskip(
    "opentelemetry.instrumentation.openai_v2",
    reason="opentelemetry-instrumentation-openai-v2 not installed",
)

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)
from opentelemetry.instrumentation.openai_v2.patch import ChoiceBuffer

from llama_stack.providers.utils.inference.stream_utils import _normalize_tool_call_arguments


class TestOTELOpenAIV2ToolCallCrash:
    """Proves that opentelemetry-instrumentation-openai-v2's ChoiceBuffer crashes
    when a provider sends ``arguments=None`` on a tool-call delta chunk.

    This is the root cause of streaming tool-call failures when llama-stack is
    launched with ``opentelemetry-instrument``.
    """

    def test_arguments_none_crashes_on_join(self):
        """The real OTEL ChoiceBuffer crashes when cleanup serializes tool-call arguments.

        Sequence:
          1. Provider sends first tool-call delta with name only (arguments=None)
          2. ChoiceBuffer.append_tool_call() appends None to the arguments list
          3. Provider sends argument fragments (strings)
          4. At stream end, cleanup() calls "".join(buffer.arguments) → TypeError

        This TypeError replaces StopAsyncIteration in the OTEL stream wrapper,
        killing the async iteration.  llama-stack's orchestrator catches it,
        emits response.failed, and skips conversation storage.
        """
        buf = ChoiceBuffer(index=0)

        # Chunk 1: tool call starts — name set, arguments=None (non-OpenAI provider)
        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc123",
                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=None),
                type="function",
            )
        )

        # Chunk 2: argument fragment streams in
        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='{"location":'),
            )
        )

        # Chunk 3: argument fragment continues
        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments=' "NYC"}'),
            )
        )

        # Verify None is in the buffer (the root cause)
        assert buf.tool_calls_buffers[0].arguments == [None, '{"location":', ' "NYC"}']

        # This is the exact line that crashes in cleanup():
        #   LegacyChatStreamWrapper.cleanup()  →  "".join(tool_call.arguments)
        #   ChatStreamWrapper._set_output_messages()  →  "".join(tool_call.arguments)
        with pytest.raises(TypeError, match="sequence item 0: expected str instance, NoneType found"):
            "".join(buf.tool_calls_buffers[0].arguments)

    def test_normalize_then_otel_does_not_crash(self):
        """After normalizing arguments=None → "", the real OTEL ChoiceBuffer works."""
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=None),
                                type="function",
                            )
                        ]
                    ),
                    finish_reason=None,
                )
            ],
            created=0,
            model="test",
            object="chat.completion.chunk",
        )

        assert chunk.choices[0].delta.tool_calls[0].function.arguments is None

        _normalize_tool_call_arguments(chunk)

        assert chunk.choices[0].delta.tool_calls[0].function.arguments == ""

        # Now feed the normalized chunk to the real OTEL ChoiceBuffer — no crash
        buf = ChoiceBuffer(index=0)
        buf.append_tool_call(chunk.choices[0].delta.tool_calls[0])
        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='{"location": "NYC"}'),
            )
        )

        result = "".join(buf.tool_calls_buffers[0].arguments)
        assert result == '{"location": "NYC"}'

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "opentelemetry-instrumentation-openai-v2 crashes on arguments=None. "
            "When this xfail starts failing (upstream fixed the bug), remove "
            "_normalize_tool_call_arguments from stream_utils.py and openai_mixin.py."
        ),
    )
    def test_arguments_none_without_normalization_crashes(self):
        """Canary: feed arguments=None to the real OTEL ChoiceBuffer WITHOUT normalization.

        This test is expected to crash (xfail).  If upstream fixes their
        ChoiceBuffer to handle None, this test will unexpectedly pass and
        strict=True will fail the suite — signaling that our normalization
        workaround can be removed.
        """
        buf = ChoiceBuffer(index=0)

        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc123",
                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=None),
                type="function",
            )
        )
        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='{"location": "NYC"}'),
            )
        )

        # This should crash — if it doesn't, upstream fixed the bug
        "".join(buf.tool_calls_buffers[0].arguments)

    def test_arguments_empty_string_does_not_crash(self):
        """Standard OpenAI format (arguments="") does not crash.

        This confirms the crash only affects non-OpenAI providers that send
        arguments=None.  OpenAI itself always sends an empty string.
        """
        buf = ChoiceBuffer(index=0)

        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                id="call_abc123",
                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
                type="function",
            )
        )

        buf.append_tool_call(
            ChoiceDeltaToolCall(
                index=0,
                function=ChoiceDeltaToolCallFunction(arguments='{"location": "NYC"}'),
            )
        )

        # No None in the buffer — join works fine
        result = "".join(buf.tool_calls_buffers[0].arguments)
        assert result == '{"location": "NYC"}'
