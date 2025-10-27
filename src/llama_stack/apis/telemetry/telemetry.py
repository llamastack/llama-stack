# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from llama_stack.models.llama.datatypes import Primitive
from llama_stack.schema_utils import json_schema_type, register_schema

# Add this constant near the top of the file, after the imports
DEFAULT_TTL_DAYS = 7


@json_schema_type
class SpanStatus(Enum):
    """The status of a span indicating whether it completed successfully or with an error.
    :cvar OK: Span completed successfully without errors
    :cvar ERROR: Span completed with an error or failure
    """

    OK = "ok"
    ERROR = "error"


@json_schema_type
class Span(BaseModel):
    """A span representing a single operation within a trace.
    :param span_id: Unique identifier for the span
    :param trace_id: Unique identifier for the trace this span belongs to
    :param parent_span_id: (Optional) Unique identifier for the parent span, if this is a child span
    :param name: Human-readable name describing the operation this span represents
    :param start_time: Timestamp when the operation began
    :param end_time: (Optional) Timestamp when the operation finished, if completed
    :param attributes: (Optional) Key-value pairs containing additional metadata about the span
    """

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    name: str
    start_time: datetime
    end_time: datetime | None = None
    attributes: dict[str, Any] | None = Field(default_factory=lambda: {})

    def set_attribute(self, key: str, value: Any):
        if self.attributes is None:
            self.attributes = {}
        self.attributes[key] = value


@json_schema_type
class Trace(BaseModel):
    """A trace representing the complete execution path of a request across multiple operations.
    :param trace_id: Unique identifier for the trace
    :param root_span_id: Unique identifier for the root span that started this trace
    :param start_time: Timestamp when the trace began
    :param end_time: (Optional) Timestamp when the trace finished, if completed
    """

    trace_id: str
    root_span_id: str
    start_time: datetime
    end_time: datetime | None = None


@json_schema_type
class EventType(Enum):
    """The type of telemetry event being logged.
    :cvar UNSTRUCTURED_LOG: A simple log message with severity level
    :cvar STRUCTURED_LOG: A structured log event with typed payload data
    :cvar METRIC: A metric measurement with value and unit
    """

    UNSTRUCTURED_LOG = "unstructured_log"
    STRUCTURED_LOG = "structured_log"
    METRIC = "metric"


@json_schema_type
class LogSeverity(Enum):
    """The severity level of a log message.
    :cvar VERBOSE: Detailed diagnostic information for troubleshooting
    :cvar DEBUG: Debug information useful during development
    :cvar INFO: General informational messages about normal operation
    :cvar WARN: Warning messages about potentially problematic situations
    :cvar ERROR: Error messages indicating failures that don't stop execution
    :cvar CRITICAL: Critical error messages indicating severe failures
    """

    VERBOSE = "verbose"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    CRITICAL = "critical"


class EventCommon(BaseModel):
    """Common fields shared by all telemetry events.
    :param trace_id: Unique identifier for the trace this event belongs to
    :param span_id: Unique identifier for the span this event belongs to
    :param timestamp: Timestamp when the event occurred
    :param attributes: (Optional) Key-value pairs containing additional metadata about the event
    """

    trace_id: str
    span_id: str
    timestamp: datetime
    attributes: dict[str, Primitive] | None = Field(default_factory=lambda: {})


@json_schema_type
class UnstructuredLogEvent(EventCommon):
    """An unstructured log event containing a simple text message.
    :param type: Event type identifier set to UNSTRUCTURED_LOG
    :param message: The log message text
    :param severity: The severity level of the log message
    """

    type: Literal[EventType.UNSTRUCTURED_LOG] = EventType.UNSTRUCTURED_LOG
    message: str
    severity: LogSeverity


@json_schema_type
class MetricEvent(EventCommon):
    """A metric event containing a measured value.
    :param type: Event type identifier set to METRIC
    :param metric: The name of the metric being measured
    :param value: The numeric value of the metric measurement
    :param unit: The unit of measurement for the metric value
    """

    type: Literal[EventType.METRIC] = EventType.METRIC
    metric: str  # this would be an enum
    value: int | float
    unit: str


@json_schema_type
class MetricInResponse(BaseModel):
    """A metric value included in API responses.
    :param metric: The name of the metric
    :param value: The numeric value of the metric
    :param unit: (Optional) The unit of measurement for the metric value
    """

    metric: str
    value: int | float
    unit: str | None = None


# This is a short term solution to allow inference API to return metrics
# The ideal way to do this is to have a way for all response types to include metrics
# and all metric events logged to the telemetry API to be included with the response
# To do this, we will need to augment all response types with a metrics field.
# We have hit a blocker from stainless SDK that prevents us from doing this.
# The blocker is that if we were to augment the response types that have a data field
# in them like so
# class ListModelsResponse(BaseModel):
# metrics: Optional[List[MetricEvent]] = None
# data: List[Models]
# ...
# The client SDK will need to access the data by using a .data field, which is not
# ergonomic. Stainless SDK does support unwrapping the response type, but it
# requires that the response type to only have a single field.

# We will need a way in the client SDK to signal that the metrics are needed
# and if they are needed, the client SDK has to return the full response type
# without unwrapping it.


class MetricResponseMixin(BaseModel):
    """Mixin class for API responses that can include metrics.
    :param metrics: (Optional) List of metrics associated with the API response
    """

    metrics: list[MetricInResponse] | None = None


@json_schema_type
class StructuredLogType(Enum):
    """The type of structured log event payload.
    :cvar SPAN_START: Event indicating the start of a new span
    :cvar SPAN_END: Event indicating the completion of a span
    """

    SPAN_START = "span_start"
    SPAN_END = "span_end"


@json_schema_type
class SpanStartPayload(BaseModel):
    """Payload for a span start event.
    :param type: Payload type identifier set to SPAN_START
    :param name: Human-readable name describing the operation this span represents
    :param parent_span_id: (Optional) Unique identifier for the parent span, if this is a child span
    """

    type: Literal[StructuredLogType.SPAN_START] = StructuredLogType.SPAN_START
    name: str
    parent_span_id: str | None = None


@json_schema_type
class SpanEndPayload(BaseModel):
    """Payload for a span end event.
    :param type: Payload type identifier set to SPAN_END
    :param status: The final status of the span indicating success or failure
    """

    type: Literal[StructuredLogType.SPAN_END] = StructuredLogType.SPAN_END
    status: SpanStatus


StructuredLogPayload = Annotated[
    SpanStartPayload | SpanEndPayload,
    Field(discriminator="type"),
]
register_schema(StructuredLogPayload, name="StructuredLogPayload")


@json_schema_type
class StructuredLogEvent(EventCommon):
    """A structured log event containing typed payload data.
    :param type: Event type identifier set to STRUCTURED_LOG
    :param payload: The structured payload data for the log event
    """

    type: Literal[EventType.STRUCTURED_LOG] = EventType.STRUCTURED_LOG
    payload: StructuredLogPayload


Event = Annotated[
    UnstructuredLogEvent | MetricEvent | StructuredLogEvent,
    Field(discriminator="type"),
]
register_schema(Event, name="Event")


@json_schema_type
class EvalTrace(BaseModel):
    """A trace record for evaluation purposes.
    :param session_id: Unique identifier for the evaluation session
    :param step: The evaluation step or phase identifier
    :param input: The input data for the evaluation
    :param output: The actual output produced during evaluation
    :param expected_output: The expected output for comparison during evaluation
    """

    session_id: str
    step: str
    input: str
    output: str
    expected_output: str


@json_schema_type
class SpanWithStatus(Span):
    """A span that includes status information.
    :param status: (Optional) The current status of the span
    """

    status: SpanStatus | None = None


@json_schema_type
class QueryConditionOp(Enum):
    """Comparison operators for query conditions.
    :cvar EQ: Equal to comparison
    :cvar NE: Not equal to comparison
    :cvar GT: Greater than comparison
    :cvar LT: Less than comparison
    """

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"


@json_schema_type
class QueryCondition(BaseModel):
    """A condition for filtering query results.
    :param key: The attribute key to filter on
    :param op: The comparison operator to apply
    :param value: The value to compare against
    """

    key: str
    op: QueryConditionOp
    value: Any


class QueryTracesResponse(BaseModel):
    """Response containing a list of traces.
    :param data: List of traces matching the query criteria
    """

    data: list[Trace]


class QuerySpansResponse(BaseModel):
    """Response containing a list of spans.
    :param data: List of spans matching the query criteria
    """

    data: list[Span]


class QuerySpanTreeResponse(BaseModel):
    """Response containing a tree structure of spans.
    :param data: Dictionary mapping span IDs to spans with status information
    """

    data: dict[str, SpanWithStatus]


class MetricQueryType(Enum):
    """The type of metric query to perform.
    :cvar RANGE: Query metrics over a time range
    :cvar INSTANT: Query metrics at a specific point in time
    """

    RANGE = "range"
    INSTANT = "instant"


class MetricLabelOperator(Enum):
    """Operators for matching metric labels.
    :cvar EQUALS: Label value must equal the specified value
    :cvar NOT_EQUALS: Label value must not equal the specified value
    :cvar REGEX_MATCH: Label value must match the specified regular expression
    :cvar REGEX_NOT_MATCH: Label value must not match the specified regular expression
    """

    EQUALS = "="
    NOT_EQUALS = "!="
    REGEX_MATCH = "=~"
    REGEX_NOT_MATCH = "!~"


class MetricLabelMatcher(BaseModel):
    """A matcher for filtering metrics by label values.
    :param name: The name of the label to match
    :param value: The value to match against
    :param operator: The comparison operator to use for matching
    """

    name: str
    value: str
    operator: MetricLabelOperator = MetricLabelOperator.EQUALS


@json_schema_type
class MetricLabel(BaseModel):
    """A label associated with a metric.
    :param name: The name of the label
    :param value: The value of the label
    """

    name: str
    value: str


@json_schema_type
class MetricDataPoint(BaseModel):
    """A single data point in a metric time series.
    :param timestamp: Unix timestamp when the metric value was recorded
    :param value: The numeric value of the metric at this timestamp
    """

    timestamp: int
    value: float
    unit: str


@json_schema_type
class MetricSeries(BaseModel):
    """A time series of metric data points.
    :param metric: The name of the metric
    :param labels: List of labels associated with this metric series
    :param values: List of data points in chronological order
    """

    metric: str
    labels: list[MetricLabel]
    values: list[MetricDataPoint]


class QueryMetricsResponse(BaseModel):
    """Response containing metric time series data.
    :param data: List of metric series matching the query criteria
    """

    data: list[MetricSeries]


@runtime_checkable
class Telemetry(Protocol):
    async def log_event(
        self,
        event: Event,
        ttl_seconds: int = DEFAULT_TTL_DAYS * 86400,
    ) -> None:
        """Log an event.

        :param event: The event to log.
        :param ttl_seconds: The time to live of the event.
        """
        ...
