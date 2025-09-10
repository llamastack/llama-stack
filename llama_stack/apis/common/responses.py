# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type


class Order(Enum):
    """Sort order for paginated responses.
    :cvar asc: Ascending order
    :cvar desc: Descending order
    """

    asc = "asc"
    desc = "desc"


@json_schema_type
class PaginatedResponse(BaseModel):
    """A generic paginated response that follows a simple format.

    :param data: The list of items for the current page
    :param has_more: Whether there are more items available after this set
    :param url: The URL for accessing this list
    """

    data: list[dict[str, Any]]
    has_more: bool
    url: str | None = None
