# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from opentelemetry.trace import SpanContext, TraceFlags

from llama_stack.providers.inline.agents.meta_reference.agent_instance import ChatAgent


class FakeSpan:
    def __init__(self, trace_id: int = 123, span_id: int = 456):
        self._context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )

    def get_span_context(self):
        return self._context


@pytest.fixture
def agent_with_telemetry():
    """Create a real ChatAgent with telemetry API"""
    telemetry_api = AsyncMock()

    agent = ChatAgent(
        agent_id="test-agent",
        agent_config=Mock(),
        inference_api=Mock(),
        safety_api=Mock(),
        tool_runtime_api=Mock(),
        tool_groups_api=Mock(),
        vector_io_api=Mock(),
        telemetry_api=telemetry_api,
        persistence_store=Mock(),
        created_at="2025-01-01T00:00:00Z",
        policy=[],
    )
    return agent


@pytest.fixture
def agent_without_telemetry():
    """Create a real ChatAgent without telemetry API"""
    agent = ChatAgent(
        agent_id="test-agent",
        agent_config=Mock(),
        inference_api=Mock(),
        safety_api=Mock(),
        tool_runtime_api=Mock(),
        tool_groups_api=Mock(),
        vector_io_api=Mock(),
        telemetry_api=None,
        persistence_store=Mock(),
        created_at="2025-01-01T00:00:00Z",
        policy=[],
    )
    return agent


class TestAgentMetrics:
    def test_step_execution_metrics(self, agent_with_telemetry, monkeypatch):
        """Test that step execution metrics are emitted correctly"""
        fake_span = FakeSpan()
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        # Capture the metric instead of actually creating async task
        captured_metrics = []

        async def capture_metric(metric):
            captured_metrics.append(metric)

        monkeypatch.setattr(agent_with_telemetry.telemetry_api, "log_event", capture_metric)

        def mock_create_task(coro, *, name=None):
            return asyncio.run(coro)

        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.asyncio.create_task", mock_create_task
        )

        agent_with_telemetry._track_step()

        assert len(captured_metrics) == 1
        metric = captured_metrics[0]
        assert metric.metric == "llama_stack_agent_steps_total"
        assert metric.value == 1
        assert metric.unit == "1"
        assert metric.attributes["agent_id"] == "test-agent"

    def test_workflow_completion_metrics(self, agent_with_telemetry, monkeypatch):
        """Test that workflow completion metrics are emitted correctly"""
        fake_span = FakeSpan()
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        captured_metrics = []

        async def capture_metric(metric):
            captured_metrics.append(metric)

        monkeypatch.setattr(agent_with_telemetry.telemetry_api, "log_event", capture_metric)

        def mock_create_task(coro, *, name=None):
            return asyncio.run(coro)

        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.asyncio.create_task", mock_create_task
        )

        agent_with_telemetry._track_workflow("completed", 2.5)

        assert len(captured_metrics) == 2

        # Check workflow count metric
        count_metric = captured_metrics[0]
        assert count_metric.metric == "llama_stack_agent_workflows_total"
        assert count_metric.value == 1
        assert count_metric.attributes["status"] == "completed"

        # Check duration metric
        duration_metric = captured_metrics[1]
        assert duration_metric.metric == "llama_stack_agent_workflow_duration_seconds"
        assert duration_metric.value == 2.5
        assert duration_metric.unit == "s"

    def test_tool_usage_metrics(self, agent_with_telemetry, monkeypatch):
        """Test that tool usage metrics are emitted correctly"""
        fake_span = FakeSpan()
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        captured_metrics = []

        async def capture_metric(metric):
            captured_metrics.append(metric)

        monkeypatch.setattr(agent_with_telemetry.telemetry_api, "log_event", capture_metric)

        def mock_create_task(coro, *, name=None):
            return asyncio.run(coro)

        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.asyncio.create_task", mock_create_task
        )

        agent_with_telemetry._track_tool("web_search")

        assert len(captured_metrics) == 1
        metric = captured_metrics[0]
        assert metric.metric == "llama_stack_agent_tool_calls_total"
        assert metric.attributes["tool"] == "web_search"

    def test_knowledge_search_tool_mapping(self, agent_with_telemetry, monkeypatch):
        """Test that knowledge_search tool is mapped to rag"""
        fake_span = FakeSpan()
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: fake_span
        )

        captured_metrics = []

        async def capture_metric(metric):
            captured_metrics.append(metric)

        monkeypatch.setattr(agent_with_telemetry.telemetry_api, "log_event", capture_metric)

        def mock_create_task(coro, *, name=None):
            return asyncio.run(coro)

        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.asyncio.create_task", mock_create_task
        )

        agent_with_telemetry._track_tool("knowledge_search")

        assert len(captured_metrics) == 1
        metric = captured_metrics[0]
        assert metric.attributes["tool"] == "rag"

    def test_no_telemetry_api(self, agent_without_telemetry):
        """Test that methods work gracefully when telemetry_api is None"""
        # These should not crash
        agent_without_telemetry._track_step()
        agent_without_telemetry._track_workflow("failed", 1.0)
        agent_without_telemetry._track_tool("web_search")

    def test_no_active_span(self, agent_with_telemetry, monkeypatch):
        """Test that methods work gracefully when no span is active"""
        monkeypatch.setattr(
            "llama_stack.providers.inline.agents.meta_reference.agent_instance.get_current_span", lambda: None
        )

        # These should not crash and should not call telemetry
        agent_with_telemetry._track_step()
        agent_with_telemetry._track_workflow("failed", 1.0)
        agent_with_telemetry._track_tool("web_search")

        # Telemetry should not have been called
        agent_with_telemetry.telemetry_api.log_event.assert_not_called()
