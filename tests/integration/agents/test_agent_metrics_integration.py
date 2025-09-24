# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import Any

from llama_stack.providers.utils.telemetry import tracing as telemetry_tracing


class TestAgentMetricsIntegration:
    async def test_agent_metrics_end_to_end(
        self: Any,
        telemetry: Any,
        kvstore: Any,
        make_agent_fixture: Any,
        span_patch: Any,
        make_completion_fn: Any,
    ) -> None:
        from llama_stack.apis.inference import (
            SamplingParams,
            UserMessage,
        )

        agent: Any = make_agent_fixture(telemetry, kvstore)

        session_id = await agent.create_session("s")
        sampling_params = SamplingParams(max_tokens=64)

        # single trace: plain, knowledge_search, web_search
        await telemetry_tracing.start_trace("agent_metrics")
        agent.inference_api.chat_completion = make_completion_fn(None, "Hello! I can help you with that.")
        async for _ in agent.run(
            session_id,
            "t1",
            [UserMessage(content="Hello")],
            sampling_params,
            stream=True,
        ):
            pass
        agent.inference_api.chat_completion = make_completion_fn("knowledge_search", "")
        async for _ in agent.run(
            session_id,
            "t2",
            [UserMessage(content="Please search knowledge")],
            sampling_params,
            stream=True,
        ):
            pass
        agent.inference_api.chat_completion = make_completion_fn("web_search", "")
        async for _ in agent.run(
            session_id,
            "t3",
            [UserMessage(content="Please search web")],
            sampling_params,
            stream=True,
        ):
            pass
        await telemetry_tracing.end_trace()

        # Poll briefly to avoid flake with async persistence
        tool_labels: set[str] = set()
        for _ in range(10):
            resp = await telemetry.query_metrics("llama_stack_agent_tool_calls_total", start_time=0, end_time=None)
            tool_labels.clear()
            for series in getattr(resp, "data", []) or []:
                for lbl in getattr(series, "labels", []) or []:
                    name = getattr(lbl, "name", None) or getattr(lbl, "key", None)
                    value = getattr(lbl, "value", None)
                    if name == "tool" and value:
                        tool_labels.add(value)

            # Look for both web_search AND some form of knowledge search
            if ("web_search" in tool_labels) and ("rag" in tool_labels or "knowledge_search" in tool_labels):
                break
            await asyncio.sleep(0.1)

        # More descriptive assertion
        assert bool(tool_labels & {"web_search", "rag", "knowledge_search"}), (
            f"Expected tool calls not found. Got: {tool_labels}"
        )
