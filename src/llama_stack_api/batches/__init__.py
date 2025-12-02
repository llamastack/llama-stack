# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Batches API protocol and models.

This module contains the Batches protocol definition.
Pydantic models are defined in llama_stack_api.batches.models.
The FastAPI router is defined in llama_stack_api.batches.fastapi_routes.
"""

try:
    from openai.types import Batch as BatchObject
except ImportError as e:
    raise ImportError("OpenAI package is required for batches API. Please install it with: pip install openai") from e

# Import protocol for re-export
from llama_stack_api.batches.api import Batches

# Import models for re-export
from llama_stack_api.batches.models import (
    CancelBatchRequest,
    CreateBatchRequest,
    ListBatchesRequest,
    ListBatchesResponse,
    RetrieveBatchRequest,
)

__all__ = [
    "Batches",
    "BatchObject",
    "CreateBatchRequest",
    "ListBatchesRequest",
    "RetrieveBatchRequest",
    "CancelBatchRequest",
    "ListBatchesResponse",
]
