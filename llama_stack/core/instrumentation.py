# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Protocol for instrumentation providers."""

from abc import abstractmethod

from fastapi import FastAPI
from pydantic import BaseModel, Field


class InstrumentationProvider(BaseModel):
    """
    Base class for instrumentation providers.

    Instrumentation providers add observability (tracing, metrics, logs) to the
    application but don't expose API endpoints.
    """

    provider: str = Field(description="Provider identifier for discriminated unions")
    config: BaseModel |  None = Field(default=None, description="Optional configuration for the instrumentation provider. Most support configuration via environment variables.")

    @abstractmethod
    def fastapi_middleware(self, app: FastAPI) -> None:
        """
        Inject middleware into the FastAPI application.

        :param app: The FastAPI application to instrument
        """
        ...
