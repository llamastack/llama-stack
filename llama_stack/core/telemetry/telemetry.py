# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import abstractmethod

from fastapi import FastAPI
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
