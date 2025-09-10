# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel

from llama_stack.apis.inference import Message
from llama_stack.schema_utils import json_schema_type, webmethod


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
    """Request to generate synthetic data. A small batch of prompts and a filtering function

    :param dialogs: List of conversation messages to use as input for synthetic data generation
    :param filtering_function: Type of filtering to apply to generated synthetic data samples
    :param model: (Optional) The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint
    """

    dialogs: list[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    model: str | None = None


@json_schema_type
class SyntheticDataGenerationResponse(BaseModel):
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold.

    :param synthetic_data: List of generated synthetic data samples that passed the filtering criteria
    :param statistics: (Optional) Statistical information about the generation process and filtering results
    """

    synthetic_data: list[dict[str, Any]]
    statistics: dict[str, Any] | None = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic-data-generation/generate")
    def synthetic_data_generate(
        self,
        dialogs: list[Message],
        filtering_function: FilteringFunction = FilteringFunction.none,
        model: str | None = None,
    ) -> SyntheticDataGenerationResponse:
        """Generate synthetic data based on input dialogs and apply filtering.

        :param dialogs: List of conversation messages to use as input for synthetic data generation
        :param filtering_function: Type of filtering to apply to generated synthetic data samples
        :param model: (Optional) The identifier of the model to use. The model must be registered with Llama Stack and available via the /models endpoint
        :returns: Response containing filtered synthetic data samples and optional statistics
        """
        ...
