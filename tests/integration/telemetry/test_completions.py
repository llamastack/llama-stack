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

    chunk_count = None
    for span in spans:
        if span.attributes.get("__type__") == "async_generator":
            chunk_count = span.attributes.get("chunk_count")
            if chunk_count:
                chunk_count = int(chunk_count)
                break

    assert chunk_count is not None
    assert chunk_count == len(chunks)


def test_telemetry_format_completeness(mock_otlp_collector, llama_stack_client, text_model_id):
    """Comprehensive validation of telemetry data format including spans and metrics."""
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai with temperature 0.7"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )

    # Handle both dict and Pydantic model for usage
    # This occurs due to the replay system returning a dict for usage, but the client returning a Pydantic model
    # TODO: Fix this by making the replay system return a Pydantic model for usage
    usage = response.usage if isinstance(response.usage, dict) else response.usage.model_dump()
    assert usage.get("prompt_tokens") and usage["prompt_tokens"] > 0
    assert usage.get("completion_tokens") and usage["completion_tokens"] > 0
    assert usage.get("total_tokens") and usage["total_tokens"] > 0

    # Verify spans
    spans = mock_otlp_collector.get_spans()
    # Expected spans: 1 root span + 3 autotraced method calls from routing/inference
    assert len(spans) == 4, f"Expected 4 spans, got {len(spans)}"

    # Collect all model_ids found in spans
    logged_model_ids = []

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
                logged_model_ids.append(args["model_id"])

    # At least one span should capture the fully qualified model ID
    assert text_model_id in logged_model_ids, f"Expected to find {text_model_id} in spans, but got {logged_model_ids}"

    # TODO: re-enable this once metrics get fixed
    """
    # Verify token usage metrics in response
    metrics = mock_otlp_collector.get_metrics()

    assert metrics
    for metric in metrics:
        assert metric.name in ["completion_tokens", "total_tokens", "prompt_tokens"]
        assert metric.unit == "tokens"
        assert metric.data.data_points and len(metric.data.data_points) == 1
        match metric.name:
            case "completion_tokens":
                assert metric.data.data_points[0].value == usage["completion_tokens"]
            case "total_tokens":
                assert metric.data.data_points[0].value == usage["total_tokens"]
            case "prompt_tokens":
                assert metric.data.data_points[0].value == usage["prompt_tokens"
    """
