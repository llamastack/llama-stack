# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


class Order(Enum):
    """Sort order for paginated responses."""

    asc = "asc"
    desc = "desc"


@json_schema_type
class PaginatedResponse(BaseModel):
    """A generic paginated response that follows a simple format."""

    data: list[dict[str, Any]] = Field(description="The list of items for the current page.")
    has_more: bool = Field(description="Whether there are more items available after this set.")
    url: str | None = Field(description="The URL for accessing this list.")
