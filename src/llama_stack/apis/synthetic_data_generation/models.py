# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from llama_stack.apis.inference import Message
from llama_stack.schema_utils import json_schema_type


class FilteringFunction(Enum):
    """The type of filtering function.

    :cvar none: No filtering applied, accept all generated synthetic data
    :cvar random: Random sampling of generated data points
    :cvar top_k: Keep only the top-k highest scoring synthetic data samples
    :cvar top_p: Nucleus-style filtering, keep samples exceeding cumulative score threshold
    :cvar top_k_top_p: Combined top-k and top-p filtering strategy
    :cvar sigmoid: Apply sigmoid function for probability-based filtering
    """

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"


@json_schema_type
class SyntheticDataGenerationRequest(BaseModel):
    """Request to generate synthetic data. A small batch of prompts and a filtering function."""

    dialogs: list[Message] = Field(
        ..., description="List of conversation messages to use as input for synthetic data generation"
    )
    filtering_function: FilteringFunction = Field(
        default=FilteringFunction.none, description="Type of filtering to apply to generated synthetic data samples"
    )
    model: str | None = Field(
        default=None,
        description="The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint",
    )


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: list[dict[str, Any]] = Field(
        ..., description="List of generated synthetic data samples that passed the filtering criteria"
    )
    statistics: dict[str, Any] | None = Field(
        default=None, description="Statistical information about the generation process and filtering results"
    )
