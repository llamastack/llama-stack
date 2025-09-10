## Telemetry

The Llama Stack telemetry system provides comprehensive tracing, metrics, and logging capabilities. It supports multiple sink types including OpenTelemetry, SQLite, and Console output.

### Events
The telemetry system supports three main types of events:

- **Unstructured Log Events**: Free-form log messages with severity levels
```python
unstructured_log_event = UnstructuredLogEvent(
    message="This is a log message", severity=LogSeverity.INFO
)
```
- **Metric Events**: Numerical measurements with units
```python
metric_event = MetricEvent(metric="my_metric", value=10, unit="count")
```
- **Structured Log Events**: System events like span start/end. Extensible to add more structured log types.
```python
structured_log_event = SpanStartPayload(name="my_span", parent_span_id="parent_span_id")
```

### Spans and Traces
- **Spans**: Represent operations with timing and hierarchical relationships
- **Traces**: Collection of related spans forming a complete request flow

### Metrics

Llama Stack automatically generates metrics during inference operations. These metrics are aggregated at the **inference request level** and provide insights into token usage and model performance.

#### Available Metrics

The following metrics are automatically generated for each inference request:

| Metric Name | Type | Unit | Description | Labels |
|-------------|------|------|-------------|--------|
| `llama_stack_prompt_tokens_total` | Counter | `tokens` | Number of tokens in the input prompt | `model_id`, `provider_id` |
| `llama_stack_completion_tokens_total` | Counter | `tokens` | Number of tokens in the generated response | `model_id`, `provider_id` |
| `llama_stack_tokens_total` | Counter | `tokens` | Total tokens used (prompt + completion) | `model_id`, `provider_id` |

#### Metric Generation Flow

1. **Token Counting**: During inference operations (chat completion, completion, etc.), the system counts tokens in both input prompts and generated responses
2. **Metric Construction**: For each request, `MetricEvent` objects are created with the token counts
3. **Telemetry Logging**: Metrics are sent to the configured telemetry sinks
4. **OpenTelemetry Export**: When OpenTelemetry is enabled, metrics are exposed as standard OpenTelemetry counters

#### Metric Aggregation Level

All metrics are generated and aggregated at the **inference request level**. This means:

- Each individual inference request generates its own set of metrics
- Metrics are not pre-aggregated across multiple requests
- Aggregation (sums, averages, etc.) can be performed by your observability tools (Prometheus, Grafana, etc.)
- Each metric includes labels for `model_id` and `provider_id` to enable filtering and grouping

#### Example Metric Event

```python
MetricEvent(
    trace_id="1234567890abcdef",
    span_id="abcdef1234567890",
    metric="total_tokens",
    value=150,
    timestamp=1703123456.789,
    unit="tokens",
    attributes={"model_id": "meta-llama/Llama-3.2-3B-Instruct", "provider_id": "tgi"},
)
```

#### Querying Metrics

When using the OpenTelemetry sink, metrics are exposed in standard OpenTelemetry format and can be queried through:

- **Prometheus**: Scrape metrics from the OpenTelemetry Collector's metrics endpoint
- **Grafana**: Create dashboards using Prometheus as a data source
- **OpenTelemetry Collector**: Forward metrics to other observability systems

Example Prometheus queries:
```promql
# Total tokens used across all models
sum(llama_stack_tokens_total)

# Tokens per model
sum by (model_id) (llama_stack_tokens_total)

# Average tokens per request
rate(llama_stack_tokens_total[5m])
```

### Sinks
- **OpenTelemetry**: Send events to an OpenTelemetry Collector. This is useful for visualizing traces in a tool like Jaeger and collecting metrics for Prometheus.
- **SQLite**: Store events in a local SQLite database. This is needed if you want to query the events later through the Llama Stack API.
- **Console**: Print events to the console.

### Providers

#### Meta-Reference Provider
Currently, only the meta-reference provider is implemented. It can be configured to send events to multiple sink types:
1) OpenTelemetry Collector (traces and metrics)
2) SQLite (traces only)
3) Console (all events)

#### Configuration

Here's an example that sends telemetry signals to all sink types. Your configuration might use only one or a subset.

```yaml
  telemetry:
  - provider_id: meta-reference
    provider_type: inline::meta-reference
    config:
      service_name: "llama-stack-service"
      sinks: ['console', 'sqlite', 'otel_trace', 'otel_metric']
      otel_exporter_otlp_endpoint: "http://localhost:4318"
      sqlite_db_path: "/path/to/telemetry.db"
```

**Environment Variables:**
- `OTEL_EXPORTER_OTLP_ENDPOINT`: OpenTelemetry Collector endpoint (default: `http://localhost:4318`)
- `OTEL_SERVICE_NAME`: Service name for telemetry (default: empty string)
- `TELEMETRY_SINKS`: Comma-separated list of sinks (default: `console,sqlite`)

### Jaeger to visualize traces

The `otel_trace` sink works with any service compatible with the OpenTelemetry collector. Traces and metrics use separate endpoints but can share the same collector.

Start a Jaeger instance with the OTLP HTTP endpoint at 4318 and the Jaeger UI at 16686 using the following command:

```bash
$ docker run --pull always --rm --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/jaeger:2.1.0
```

Once the Jaeger instance is running, you can visualize traces by navigating to http://localhost:16686/.

### Querying Traces Stored in SQLite

The `sqlite` sink allows you to query traces without an external system. Here are some example
queries. Refer to the notebook at [Llama Stack Building AI
Applications](https://github.com/meta-llama/llama-stack/blob/main/docs/getting_started.ipynb) for
more examples on how to query traces and spans.
