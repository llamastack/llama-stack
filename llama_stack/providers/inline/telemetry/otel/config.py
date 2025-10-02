from typing import Literal

from pydantic import BaseModel, Field


type BatchSpanProcessor = Literal["batch"]
type SimpleSpanProcessor = Literal["simple"]


class OTelTelemetryConfig(BaseModel):
    """
    The configuration for the OpenTelemetry telemetry provider.
    Most configuration is set using environment variables.
    See https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/ for more information.
    """
    service_name: str = Field(
        description="""The name of the service to be monitored. 
        Is overridden by the OTEL_SERVICE_NAME or OTEL_RESOURCE_ATTRIBUTES environment variables.""",
    )
    service_version: str | None = Field(
        description="""The version of the service to be monitored. 
        Is overriden by the OTEL_RESOURCE_ATTRIBUTES environment variable."""
    )
    deployment_environment: str | None = Field(
        description="""The name of the environment of the service to be monitored. 
        Is overriden by the OTEL_RESOURCE_ATTRIBUTES environment variable."""
    )
    span_processor: BatchSpanProcessor | SimpleSpanProcessor | None = Field(
        description="""The span processor to use. 
        Is overriden by the OTEL_SPAN_PROCESSOR environment variable.""",
        default="batch"
    )
