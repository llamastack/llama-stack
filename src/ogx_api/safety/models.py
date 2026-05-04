# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

from ogx_api.schema_utils import json_schema_type


@json_schema_type
class RunModerationRequest(BaseModel):
    """Request model for running content moderation."""

    input: str | list[str] = Field(
        ...,
        description="Input (or inputs) to classify. Can be a single string or an array of strings.",
    )
    model: str | None = Field(
        None,
        description="The content moderation model to use. If not specified, the default shield will be used.",
    )


__all__ = [
    "RunModerationRequest",
]
