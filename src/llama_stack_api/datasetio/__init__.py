# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""DatasetIO API protocol and models.

This module contains the DatasetIO protocol definition.
Pydantic models are defined in llama_stack_api.datasetio.models.
The FastAPI router is defined in llama_stack_api.datasetio.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for FastAPI router
from .api import DatasetIO, DatasetStore

# Import models for re-export
from .models import (
    AppendRowsRequest,
    IterRowsRequest,
    PaginatedResponse,
)

__all__ = [
    "DatasetIO",
    "DatasetStore",
    "AppendRowsRequest",
    "IterRowsRequest",
    "PaginatedResponse",
    "fastapi_routes",
]
