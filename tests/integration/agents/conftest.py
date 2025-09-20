# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from llama_stack.apis.inference import ToolDefinition
from llama_stack.apis.tools import ToolInvocationResult
from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent
from llama_stack.providers.inline.telemetry.meta_reference.config import (
    TelemetryConfig,
    TelemetrySink,
)
from llama_stack.providers.inline.telemetry.meta_reference.telemetry import (
    TelemetryAdapter,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
from llama_stack.providers.utils.kvstore.sqlite.sqlite import SqliteKVStoreImpl
from llama_stack.providers.utils.telemetry import tracing as telemetry_tracing


@pytest.fixture
def make_agent_fixture():
    def _make(telemetry, kvstore) -> ChatAgent:
        agent = ChatAgent(
            agent_id="test-agent",
            agent_config=Mock(),
            inference_api=Mock(),
            safety_api=Mock(),
            tool_runtime_api=Mock(),
            tool_groups_api=Mock(),
            vector_io_api=Mock(),
            telemetry_api=telemetry,
            persistence_store=kvstore,
            created_at="2025-01-01T00:00:00Z",
            policy=[],
        )
        agent.agent_config.client_tools = []
        agent.agent_config.max_infer_iters = 5
        agent.input_shields = []
        agent.output_shields = []
        agent.tool_defs = [
            ToolDefinition(tool_name="web_search", description="", parameters={}),
            ToolDefinition(tool_name="knowledge_search", description="", parameters={}),
        ]
        agent.tool_name_to_args = {}

        # Stub tool runtime invoke_tool
        async def _mock_invoke_tool(
            *args: Any,
            tool_name: str | None = None,
            kwargs: dict | None = None,
            **extra: Any,
        ):
            return ToolInvocationResult(content="Tool execution result")

        agent.tool_runtime_api.invoke_tool = _mock_invoke_tool
        return agent

    return _make


def _chat_stream(tool_name: str | None, content: str = ""):
    from llama_stack.apis.common.content_types import (
        TextDelta,
        ToolCallDelta,
        ToolCallParseStatus,
    )
    from llama_stack.apis.inference import (
        ChatCompletionResponseEvent,
        ChatCompletionResponseEventType,
        ChatCompletionResponseStreamChunk,
        StopReason,
    )
    from llama_stack.models.llama.datatypes import ToolCall

    async def gen():
        # Start
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta=TextDelta(text=""),
            )
        )

        # Content
        if content:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=TextDelta(text=content),
                )
            )

        # Tool call if specified
        if tool_name:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        tool_call=ToolCall(call_id="call_0", tool_name=tool_name, arguments={}),
                        parse_status=ToolCallParseStatus.succeeded,
                    ),
                )
            )

        # Complete
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.complete,
                delta=TextDelta(text=""),
                stop_reason=StopReason.end_of_turn,
            )
        )

    return gen()


@pytest.fixture
async def telemetry(tmp_path: Path) -> AsyncGenerator[TelemetryAdapter, None]:
    db_path = tmp_path / "trace_store.db"
    cfg = TelemetryConfig(
        sinks=[TelemetrySink.CONSOLE, TelemetrySink.SQLITE],
        sqlite_db_path=str(db_path),
    )
    telemetry = TelemetryAdapter(cfg, deps={})
    telemetry_tracing.setup_logger(telemetry)
    try:
        yield telemetry
    finally:
        await telemetry.shutdown()


@pytest.fixture
async def kvstore(tmp_path: Path) -> SqliteKVStoreImpl:
    kv_path = tmp_path / "agent_kvstore.db"
    kv = SqliteKVStoreImpl(SqliteKVStoreConfig(db_path=str(kv_path)))
    await kv.initialize()
    return kv


@pytest.fixture
def span_patch():
    with (
        patch("llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span") as mock_span,
        patch(
            "llama_stack.providers.utils.telemetry.tracing.generate_span_id",
            return_value="0000000000000abc",
        ),
    ):
        mock_span.return_value = Mock(get_span_context=Mock(return_value=Mock(trace_id=0x123, span_id=0xABC)))
        yield


@pytest.fixture
def make_completion_fn() -> Callable[[str | None, str], Callable]:
    def _factory(tool_name: str | None = None, content: str = "") -> Callable:
        async def chat_completion(*args: Any, **kwargs: Any):
            return _chat_stream(tool_name, content)

        return chat_completion

    return _factory
