# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
OpenAI-compatible filter types for vector store search operations.

These filter types allow filtering search results based on metadata fields,
supporting both comparison operations (eq, ne, gt, etc.) and compound
operations (and, or) for complex filtering logic.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class ComparisonFilter(BaseModel):
    """A filter that compares a metadata field against a value.

    :param type: The comparison operator to apply
    :param key: The metadata field name to filter on
    :param value: The value to compare against
    """

    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"]
    key: str
    value: Any


class CompoundFilter(BaseModel):
    """A filter that combines multiple filters with a logical operator.

    :param type: The logical operator ("and" requires all filters match, "or" requires any filter matches)
    :param filters: The list of filters to combine
    """

    type: Literal["and", "or"]
    filters: list[Filter]


# Discriminated union of all filter types
Filter = Annotated[
    ComparisonFilter | CompoundFilter,
    Field(discriminator="type"),
]
