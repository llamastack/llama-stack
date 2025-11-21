# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Utilities for creating FastAPI routers with standard error responses.

This module provides standard error response definitions for FastAPI routers.
These responses use OpenAPI $ref references to component responses defined
in the OpenAPI specification.
"""

from typing import Any

standard_responses: dict[int | str, dict[str, Any]] = {
    400: {"$ref": "#/components/responses/BadRequest400"},
    429: {"$ref": "#/components/responses/TooManyRequests429"},
    500: {"$ref": "#/components/responses/InternalServerError500"},
    "default": {"$ref": "#/components/responses/DefaultError"},
}
