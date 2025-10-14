# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry tests verifying @trace_protocol decorator format using in-memory exporter."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LLAMA_STACK_TEST_STACK_CONFIG_TYPE") == "server",
    reason="In-memory telemetry tests only work in library_client mode (server mode runs in separate process)",
)


def test_streaming_chunk_count(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify streaming adds chunk_count and __type__=async_generator."""

    stream = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai 1"}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0

    spans = mock_otlp_collector.get_spans()
    assert len(spans) > 0

    for span in spans:
        if span.attributes.get("__type__") == "async_generator":
            chunk_count = span.attributes.get("chunk_count")
            if chunk_count:
                assert int(chunk_count) == len(chunks)


def test_telemetry_format_completeness(mock_otlp_collector, llama_stack_client, text_model_id):
    """Comprehensive validation of telemetry data format including spans and metrics."""
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai with temperature 0.7"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )

    assert response.usage.get("prompt_tokens") > 0
    assert response.usage.get("completion_tokens") > 0
    assert response.usage.get("total_tokens") > 0

    # Verify spans
    spans = mock_otlp_collector.get_spans()
    assert len(spans) == 5

    for span in spans:
        attrs = span.attributes
        assert attrs is not None

        # Root span is created manually by tracing middleware, not by @trace_protocol decorator
        is_root_span = attrs.get("__root__") is True

        if is_root_span:
            # Root spans have different attributes
            assert attrs.get("__location__") in ["library_client", "server"]
        else:
            # Non-root spans are created by @trace_protocol decorator
            assert attrs.get("__autotraced__")
            assert attrs.get("__class__") and attrs.get("__method__")
            assert attrs.get("__type__") in ["async", "sync", "async_generator"]

            args = json.loads(attrs["__args__"])
            if "model_id" in args:
                assert args.get("model_id") == text_model_id
            else:
                assert args.get("model") == text_model_id

    # Verify token usage metrics in response
    metrics = mock_otlp_collector.get_metrics()
    print(f"metrics: {metrics}")
    assert metrics
    for metric in metrics:
        assert metric.name in ["completion_tokens", "total_tokens", "prompt_tokens"]
        assert metric.unit == "tokens"
        assert metric.data.data_points and len(metric.data.data_points) == 1
        match metric.name:
            case "completion_tokens":
                assert metric.data.data_points[0].value == response.usage.get("completion_tokens")
            case "total_tokens":
                assert metric.data.data_points[0].value == response.usage.get("total_tokens")
            case "prompt_tokens":
                assert metric.data.data_points[0].value == response.usage.get("prompt_tokens")
