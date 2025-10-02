# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import abstractmethod
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any


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
    def custom_trace(self, name: str, *args, **kwargs) -> Any:
        """
        Creates a custom trace.
        """
        ...
    
    @abstractmethod
    def record_count(self, name: str, *args, **kwargs):
        """
        Increments a counter metric.
        """
        ...
    
    @abstractmethod
    def record_histogram(self, name: str, *args, **kwargs):
        """
        Records a histogram metric.
        """
        ...
    
    @abstractmethod
    def record_up_down_counter(self, name: str, *args, **kwargs):
        """
        Records an up/down counter metric.
        """
        ...
