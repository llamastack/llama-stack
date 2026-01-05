# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""File Processors API protocol and models.

This module contains the File Processors protocol definition.
Pydantic models are defined in llama_stack_api.file_processors.models.
The FastAPI router is defined in llama_stack_api.file_processors.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import protocol for re-export
from .api import FileProcessors

# Import models for re-export
from .models import ProcessFileResponse

__all__ = [
    "FileProcessors",
    "ProcessFileResponse",
    "fastapi_routes",
]
