# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

try:
    from openai.types import Batch as BatchObject
except ImportError:
    BatchObject = None  # type: ignore[assignment,misc]

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .batches_service import BatchService
from .models import CreateBatchRequest, ListBatchesResponse

# Backward compatibility - export Batches as alias for BatchService
Batches = BatchService

__all__ = ["Batches", "BatchService", "BatchObject", "ListBatchesResponse", "CreateBatchRequest"]
