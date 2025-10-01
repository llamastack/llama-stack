# Open Telemetry Native Instrumentation

This instrumentation package is simple, and follows expected open telemetry standards. It injects middleware for distributed tracing into all ingress and egress points into the application, and can be tuned and configured with OTEL environment variables.

## Set Up

First, bootstrap and install all necessary libraries for open telemtry:

```
uv run opentelemetry-bootstrap -a requirements | uv pip install --requirement -
```

Then, run with automatic code injection:

```
uv run opentelemetry-instrument llama stack run --config myconfig.yaml
```

### Excluded Fast API URLs

```
export OTEL_PYTHON_FASTAPI_EXCLUDED_URLS="client/.*/info,healthcheck"
```

#### Environment Variables
Environment Variables: https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/
