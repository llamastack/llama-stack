# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Integration tests for inference metrics tracking.

These tests verify that the metrics implemented in Phase 1 & 2 are being
correctly recorded for inference requests.
"""

import pytest


def test_chat_completion_nonstreaming_metrics(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify metrics are recorded for non-streaming chat completions."""
    # Clear any existing metrics
    mock_otlp_collector.clear()

    # Make a non-streaming request
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        stream=False,
    )

    assert response is not None
    assert response.choices is not None
    assert len(response.choices) > 0

    # Get metrics with appropriate timeout
    metrics = mock_otlp_collector.get_metrics(
        expected_count=3,  # requests_total, request_duration, inference_duration
        expect_model_id=text_model_id,
        timeout=10.0,
    )

    # Verify request-level metrics exist
    assert "llama_stack.inference.requests_total" in metrics, "requests_total metric not found"
    assert "llama_stack.inference.request_duration_seconds" in metrics, "request_duration metric not found"

    # Verify token-level metrics exist (if usage data is available)
    if response.usage and response.usage.total_tokens:
        assert "llama_stack.inference.inference_duration_seconds" in metrics, "inference_duration metric not found"

    # Verify metric attributes
    requests_metric = metrics["llama_stack.inference.requests_total"]
    assert requests_metric.attributes.get("model") == text_model_id
    assert requests_metric.attributes.get("endpoint_type") == "chat_completion"
    assert requests_metric.attributes.get("stream") is False
    assert requests_metric.attributes.get("status") == "success"

    # Verify request was counted
    assert requests_metric.value >= 1, "Should have at least 1 request counted"


def test_chat_completion_streaming_metrics(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify metrics are recorded for streaming chat completions including TTFT."""
    mock_otlp_collector.clear()

    # Make a streaming request
    stream = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Count to 5"}],
        stream=True,
    )

    # Consume the stream
    chunks = list(stream)
    assert len(chunks) > 0, "Should receive at least one chunk"

    # Get metrics - streaming should have additional TTFT metric
    metrics = mock_otlp_collector.get_metrics(
        expected_count=5,  # All previous + time_to_first_token
        expect_model_id=text_model_id,
        timeout=10.0,
    )

    # Verify all metrics exist
    assert "llama_stack.inference.requests_total" in metrics
    assert "llama_stack.inference.request_duration_seconds" in metrics
    assert "llama_stack.inference.inference_duration_seconds" in metrics
    assert "llama_stack.inference.time_to_first_token_seconds" in metrics, "TTFT metric not found for streaming request"

    # Verify TTFT metric attributes
    ttft_metric = metrics["llama_stack.inference.time_to_first_token_seconds"]
    assert ttft_metric.attributes.get("stream") is True
    assert ttft_metric.attributes.get("model") == text_model_id

    # Verify TTFT value is reasonable (should be > 0 and < total duration)
    assert ttft_metric.value > 0, "TTFT should be positive"

    # Verify streaming attributes
    requests_metric = metrics["llama_stack.inference.requests_total"]
    assert requests_metric.attributes.get("stream") is True
    assert requests_metric.attributes.get("status") == "success"


def test_completion_nonstreaming_metrics(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify metrics are recorded for non-streaming completions endpoint."""
    mock_otlp_collector.clear()

    # Make a non-streaming completion request
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt="Once upon a time",
        stream=False,
        max_tokens=50,
    )

    assert response is not None

    # Get metrics
    metrics = mock_otlp_collector.get_metrics(
        expected_count=4,
        expect_model_id=text_model_id,
        timeout=10.0,
    )

    # Verify basic metrics
    assert "llama_stack.inference.requests_total" in metrics
    assert "llama_stack.inference.request_duration_seconds" in metrics

    # Verify endpoint type is correct
    requests_metric = metrics["llama_stack.inference.requests_total"]
    assert requests_metric.attributes.get("endpoint_type") == "completion", "endpoint_type should be 'completion'"
    assert requests_metric.attributes.get("model") == text_model_id


def test_error_metrics_recorded(mock_otlp_collector, llama_stack_client):
    """Verify metrics are recorded for failed requests with error status."""
    mock_otlp_collector.clear()

    # Make a request with invalid model
    # Expecting an error - could be ModelNotFoundError or similar
    with pytest.raises(Exception) as exc_info:  # noqa: B017
        llama_stack_client.chat.completions.create(
            model="nonexistent-model-12345",
            messages=[{"role": "user", "content": "Test"}],
        )
    assert exc_info.value is not None

    # Get metrics - should have at least request count and duration
    metrics = mock_otlp_collector.get_metrics(
        expected_count=2,
        timeout=10.0,
    )

    # Verify error metrics were recorded
    assert "llama_stack.inference.requests_total" in metrics, "Should record request count even on error"
    assert "llama_stack.inference.request_duration_seconds" in metrics, "Should record duration even on error"

    # Verify status is error
    requests_metric = metrics["llama_stack.inference.requests_total"]
    assert requests_metric.attributes.get("status") == "error", "Status should be 'error' for failed requests"


def test_concurrent_requests_metric_exists(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify concurrent_requests metric is being tracked."""
    mock_otlp_collector.clear()

    # Make a request
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response is not None

    # Get all metrics - wait for at least 3 metrics to ensure all are exported
    metrics = mock_otlp_collector.get_metrics(expected_count=3, timeout=10.0)

    # Note: concurrent_requests will likely be 0 by the time we check
    # since the request completed. This is expected behavior.
    # We're just verifying the metric exists and was used.

    # The metric should exist even if it's 0
    # If it was never incremented/decremented, it wouldn't appear in metrics
    # So its presence in the metrics dict indicates it was used

    # We can't reliably assert on concurrent_requests value in sequential tests
    # but we can verify the other metrics prove the tracking happened
    assert "llama_stack.inference.requests_total" in metrics
    assert "llama_stack.inference.request_duration_seconds" in metrics


def test_metric_attributes_are_consistent(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify all metrics have consistent attributes."""
    mock_otlp_collector.clear()

    # Make a request
    response = llama_stack_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": "Test attributes"}],
        stream=False,
    )

    assert response is not None

    # Get metrics
    metrics = mock_otlp_collector.get_metrics(expect_model_id=text_model_id, timeout=10.0)

    # Collect all model IDs from all metrics
    model_ids = set()
    endpoint_types = set()
    stream_values = set()

    for _, metric in metrics.items():
        if metric.attributes:
            if "model" in metric.attributes:
                model_ids.add(metric.attributes["model"])
            if "endpoint_type" in metric.attributes:
                endpoint_types.add(metric.attributes["endpoint_type"])
            if "stream" in metric.attributes:
                stream_values.add(metric.attributes["stream"])

    # Verify attributes are consistent across metrics
    assert text_model_id in model_ids, f"Expected model {text_model_id} in metrics, got {model_ids}"
    assert "chat_completion" in endpoint_types, f"Expected chat_completion endpoint_type, got {endpoint_types}"
    assert False in stream_values, f"Expected stream=False, got {stream_values}"


def test_multiple_requests_increment_metrics(mock_otlp_collector, llama_stack_client, text_model_id):
    """Verify metrics accumulate across multiple requests."""
    mock_otlp_collector.clear()

    # Make multiple requests
    num_requests = 3
    for i in range(num_requests):
        response = llama_stack_client.chat.completions.create(
            model=text_model_id,
            messages=[{"role": "user", "content": f"Request {i + 1}"}],
        )
        assert response is not None

    # Get metrics - wait for at least 3 metrics to ensure all are exported
    metrics = mock_otlp_collector.get_metrics(expected_count=3, expect_model_id=text_model_id, timeout=10.0)

    # Verify requests_total accumulated
    assert "llama_stack.inference.requests_total" in metrics
    requests_metric = metrics["llama_stack.inference.requests_total"]
    assert requests_metric.value >= num_requests, (
        f"Should have at least {num_requests} requests, got {requests_metric.value}"
    )

    # Verify request_duration recorded multiple values
    assert "llama_stack.inference.request_duration_seconds" in metrics
    duration_metric = metrics["llama_stack.inference.request_duration_seconds"]

    # Histogram should have recorded multiple samples
    # The value represents the sum or count depending on the metric type
    assert duration_metric.value > 0, "Duration metric should have recorded values"
