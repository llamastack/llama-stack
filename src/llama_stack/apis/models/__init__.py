# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .model_schemas import (
    ListModelsResponse,
    Model,
    ModelInput,
    ModelType,
    OpenAIListModelsResponse,
    OpenAIModel,
    RegisterModelRequest,
)
from .models_service import ModelService

# Backward compatibility - export Models as alias for ModelService
Models = ModelService

__all__ = [
    "Models",
    "ModelService",
    "Model",
    "ModelInput",
    "ModelType",
    "ListModelsResponse",
    "RegisterModelRequest",
    "OpenAIModel",
    "OpenAIListModelsResponse",
]
