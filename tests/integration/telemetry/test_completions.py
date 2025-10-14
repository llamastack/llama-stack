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
    collector = mock_otlp_collector

    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test trace openai with temperature 0.7"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )

    assert response

    # Verify spans
    spans = collector.get_spans()
    assert len(spans) == 5

    for span in spans:
        print(f"Span: {span.attributes}")
        if span.attributes.get("__autotraced__"):
            assert span.attributes.get("__class__") and span.attributes.get("__method__")
            assert span.attributes.get("__type__") in ["async", "sync", "async_generator"]
        if span.attributes.get("__args__"):
            args = json.loads(span.attributes.get("__args__"))
            # The parameter is 'model' in openai_chat_completion, not 'model_id'
            if "model" in args:
                assert args.get("model") == text_model_id

    # Verify token metrics in response
    # Note: Llama Stack emits token metrics in the response JSON, not via OTel Metrics API
    usage = response.usage if hasattr(response, "usage") else response.get("usage")
    assert usage
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else usage.prompt_tokens
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else usage.completion_tokens
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else usage.total_tokens

    assert prompt_tokens is not None and prompt_tokens > 0
    assert completion_tokens is not None and completion_tokens > 0
    assert total_tokens is not None and total_tokens > 0
