# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenTelemetry metrics for llama-stack inference operations.

This module provides centralized metric definitions for tracking:
- Request-level metrics (total requests, duration, concurrency)
- Token-level metrics (tokens/second, inference duration, time to first token)

All metrics follow OpenTelemetry semantic conventions and use the llama_stack prefix
for consistent naming across the telemetry stack.
"""

# Import telemetry setup first to ensure OTLP exporter is configured
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, UpDownCounter

from . import setup_telemetry  # noqa: F401
from .constants import (
    CONCURRENT_REQUESTS,
    INFERENCE_DURATION,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TIME_TO_FIRST_TOKEN,
)

# Get or create meter for llama_stack.inference
# This uses the global MeterProvider configured by OTEL auto-instrumentation
# or set explicitly via metrics.set_meter_provider()
meter = metrics.get_meter("llama_stack.inference", version="1.0.0")


# Request-level metrics
# These track overall request patterns and server load

requests_total: Counter = meter.create_counter(
    name=REQUESTS_TOTAL,
    description="Total number of inference requests processed by the server",
    unit="1",
)

request_duration: Histogram = meter.create_histogram(
    name=REQUEST_DURATION,
    description="Duration of inference requests from start to completion",
    unit="s",
)

concurrent_requests: UpDownCounter = meter.create_up_down_counter(
    name=CONCURRENT_REQUESTS,
    description="Number of concurrent inference requests currently being processed",
    unit="1",
)


# Token-level metrics
# These track model inference performance and token generation rates

inference_duration: Histogram = meter.create_histogram(
    name=INFERENCE_DURATION,
    description="Time spent in model inference (excludes preprocessing/postprocessing)",
    unit="s",
)

time_to_first_token: Histogram = meter.create_histogram(
    name=TIME_TO_FIRST_TOKEN,
    description="Time from request start until first token is generated (streaming only)",
    unit="s",
)


# Utility function for creating metric attributes
def create_metric_attributes(
    model: str | None = None,
    provider: str | None = None,
    endpoint_type: str | None = None,
    stream: bool | None = None,
    status: str | None = None,
) -> dict[str, str | bool]:
    """Create a consistent attribute dictionary for metrics.

    Args:
        model: Model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        provider: Provider ID (e.g., "inline::meta-reference")
        endpoint_type: Type of endpoint ("chat_completion", "completion", "embeddings")
        stream: Whether request is streaming
        status: Request outcome ("success", "error")

    Returns:
        Dictionary of attributes with non-None values
    """
    attributes: dict[str, str | bool] = {}

    if model is not None:
        attributes["model"] = model
    if provider is not None:
        attributes["provider"] = provider
    if endpoint_type is not None:
        attributes["endpoint_type"] = endpoint_type
    if stream is not None:
        attributes["stream"] = stream
    if status is not None:
        attributes["status"] = status

    return attributes
