# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

from llama_stack.apis.inference import Message
from llama_stack.core.telemetry.trace_protocol import trace_protocol

from .models import FilteringFunction, SyntheticDataGenerationResponse


@runtime_checkable
@trace_protocol
class SyntheticDataGenerationService(Protocol):
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
