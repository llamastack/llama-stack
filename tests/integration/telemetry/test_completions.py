# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry tests verifying @trace_protocol decorator format across stack modes.

Note: The mock_otlp_collector fixture automatically clears telemetry data
before and after each test, ensuring test isolation.
"""

import json


def test_streaming_chunk_count(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify streaming adds chunk_count and __type__=async_generator."""
    stream = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai 1"}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0

    spans = mock_otlp_collector.get_spans(expected_count=5)
    assert len(spans) > 0

    async_generator_span = next(
        (
            span
            for span in reversed(spans)
            if span.get_span_type() == "async_generator"
            and span.get_attribute("chunk_count")
            and span.has_message("Test trace openai 1")
        ),
        None,
    )

    assert async_generator_span is not None

    raw_chunk_count = async_generator_span.get_attribute("chunk_count")
    assert raw_chunk_count is not None
    chunk_count = int(raw_chunk_count)

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
    spans = mock_otlp_collector.get_spans(expected_count=7)
    target_span = next(
        (span for span in reversed(spans) if span.has_message("Test trace openai with temperature 0.7")),
        None,
    )
    assert target_span is not None

    trace_id = target_span.get_trace_id()
    assert trace_id is not None

    spans = [span for span in spans if span.get_trace_id() == trace_id]
    spans = [span for span in spans if span.is_root_span() or span.is_autotraced()]
    assert len(spans) >= 4

    # Collect all model_ids found in spans
    logged_model_ids = []

    for span in spans:
        attrs = span.get_attributes()
        assert attrs is not None

        # Root span is created manually by tracing middleware, not by @trace_protocol decorator
        if span.is_root_span():
            assert span.get_location() in ["library_client", "server"]
            continue

        assert span.is_autotraced()
        class_name, method_name = span.get_class_method()
        assert class_name and method_name
        assert span.get_span_type() in ["async", "sync", "async_generator"]

        args_field = span.get_attribute("__args__")
        if args_field:
            args = json.loads(args_field)
            if "model_id" in args:
                logged_model_ids.append(args["model_id"])

    # At least one span should capture the fully qualified model ID
    assert text_model_id in logged_model_ids, f"Expected to find {text_model_id} in spans, but got {logged_model_ids}"

    # Verify token usage metrics in response
    # Verify expected metrics are present
    expected_metrics = ["completion_tokens", "total_tokens", "prompt_tokens"]
    for metric_name in expected_metrics:
        assert mock_otlp_collector.has_metric(metric_name), (
            f"Expected metric {metric_name} not found in {mock_otlp_collector.get_metric_names()}"
        )

    # Verify metric values match usage data
    assert mock_otlp_collector.get_metric_value("completion_tokens") == usage["completion_tokens"]
    assert mock_otlp_collector.get_metric_value("total_tokens") == usage["total_tokens"]
    assert mock_otlp_collector.get_metric_value("prompt_tokens") == usage["prompt_tokens"]
