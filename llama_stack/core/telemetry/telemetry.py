# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import abstractmethod

from fastapi import FastAPI
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Attributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer
from pydantic import BaseModel
from sqlalchemy import Engine


class TelemetryProvider(BaseModel):
    """
    TelemetryProvider standardizes how telemetry is provided to the application.
    """

    @abstractmethod
    def fastapi_middleware(self, app: FastAPI, *args, **kwargs):
        """
        Injects FastAPI middleware that instruments the application for telemetry.
        """
        ...

    @abstractmethod
    def sqlalchemy_instrumentation(self, engine: Engine | None = None):
        """
        Injects SQLAlchemy instrumentation that instruments the application for telemetry.
        """
        ...

    @abstractmethod
    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        tracer_provider: TracerProvider | None = None,
        schema_url: str | None = None,
        attributes: Attributes | None = None,
    ) -> Tracer:
        """
        Gets a tracer.
        """
        ...

    @abstractmethod
    def get_meter(
        self,
        name: str,
        version: str = "",
        meter_provider: MeterProvider | None = None,
        schema_url: str | None = None,
        attributes: Attributes | None = None,
    ) -> Meter:
        """
        Gets a meter.
        """
        ...
