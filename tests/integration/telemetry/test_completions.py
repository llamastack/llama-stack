# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry tests verifying @trace_protocol decorator format across stack modes."""

import json


def test_streaming_chunk_count(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify streaming adds chunk_count and __type__=async_generator."""
    mock_otlp_collector.clear()

    stream = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai 1"}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) > 0

    spans = mock_otlp_collector.get_spans(expected_count=5)
    assert len(spans) > 0

    spans = [s for s in spans if s.attributes.get("__type__") == "async_generator" and s.attributes.get("chunk_count")]
    for s in spans:
        print(s.attributes)

    async_generator_span = next(
        (s for s in spans if s.attributes.get("__type__") == "async_generator" and s.attributes.get("chunk_count")),
        None,
    )

    assert async_generator_span is not None

    raw_chunk_count = async_generator_span.attributes.get("chunk_count")
    assert raw_chunk_count is not None
    chunk_count = int(raw_chunk_count)

    assert chunk_count == len(chunks)


def test_telemetry_format_completeness(mock_otlp_collector, llama_stack_client, text_model_id):
    """Comprehensive validation of telemetry data format including spans and metrics."""
    mock_otlp_collector.clear()

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
    spans = [span for span in spans if span.attributes.get("__root__") or span.attributes.get("__autotraced__")]
    assert len(spans) >= 5

    # we only need this captured one time
    logged_model_id = None

    for span in spans:
        attrs = span.attributes
        assert attrs is not None

        # Root span is created manually by tracing middleware, not by @trace_protocol decorator
        is_root_span = attrs.get("__root__") is True

        if is_root_span:
            assert attrs.get("__location__") in ["library_client", "server"]
            continue

        assert attrs.get("__autotraced__")
        assert attrs.get("__class__") and attrs.get("__method__")
        assert attrs.get("__type__") in ["async", "sync", "async_generator"]

        args_field = attrs.get("__args__")
        if args_field:
            args = json.loads(args_field)
            if "model_id" in args:
                logged_model_id = args["model_id"]

    assert logged_model_id is not None
    assert logged_model_id == text_model_id

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
