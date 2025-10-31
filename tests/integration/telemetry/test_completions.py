# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Telemetry tests verifying @trace_protocol decorator format across stack modes."""

import json


def _span_attributes(span):
    attrs = getattr(span, "attributes", None)
    if attrs is None:
        return {}
    # ReadableSpan.attributes acts like a mapping
    try:
        return dict(attrs.items())  # type: ignore[attr-defined]
    except AttributeError:
        try:
            return dict(attrs)
        except TypeError:
            return attrs


def _span_attr(span, key):
    attrs = _span_attributes(span)
    return attrs.get(key)


def _span_trace_id(span):
    context = getattr(span, "context", None)
    if context and getattr(context, "trace_id", None) is not None:
        return f"{context.trace_id:032x}"
    return getattr(span, "trace_id", None)


def _span_has_message(span, text: str) -> bool:
    args = _span_attr(span, "__args__")
    if not args or not isinstance(args, str):
        return False
    return text in args


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

    async_generator_span = next(
        (
            span
            for span in reversed(spans)
            if _span_attr(span, "__type__") == "async_generator"
            and _span_attr(span, "chunk_count")
            and _span_has_message(span, "Test trace openai 1")
        ),
        None,
    )

    assert async_generator_span is not None

    raw_chunk_count = _span_attr(async_generator_span, "chunk_count")
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
    target_span = next(
        (span for span in reversed(spans) if _span_has_message(span, "Test trace openai with temperature 0.7")),
        None,
    )
    assert target_span is not None

    trace_id = _span_trace_id(target_span)
    assert trace_id is not None

    spans = [span for span in spans if _span_trace_id(span) == trace_id]
    spans = [span for span in spans if _span_attr(span, "__root__") or _span_attr(span, "__autotraced__")]
    assert len(spans) >= 4

    # Collect all model_ids found in spans
    logged_model_ids = []

    for span in spans:
        attrs = _span_attributes(span)
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
