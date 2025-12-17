# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Datasets API protocol and models.

This module contains the Datasets protocol definition.
Pydantic models are defined in llama_stack_api.datasets.models.
The FastAPI router is defined in llama_stack_api.datasets.fastapi_routes.
"""

# Import fastapi_routes for router factory access
from . import fastapi_routes

# Import new protocol for FastAPI router
from .api import Datasets

# Import models for re-export
from .models import (
    CommonDatasetFields,
    Dataset,
    DatasetPurpose,
    DatasetType,
    DataSource,
    GetDatasetRequest,
    ListDatasetsResponse,
    RegisterDatasetRequest,
    RowsDataSource,
    UnregisterDatasetRequest,
    URIDataSource,
)


# Define DatasetInput for backward compatibility
class DatasetInput(CommonDatasetFields):
    """Input parameters for dataset operations.

    :param dataset_id: Unique identifier for the dataset
    """

    dataset_id: str


__all__ = [
    "Datasets",
    "Dataset",
    "CommonDatasetFields",
    "DatasetPurpose",
    "DataSource",
    "DatasetInput",
    "DatasetType",
    "RowsDataSource",
    "URIDataSource",
    "ListDatasetsResponse",
    "RegisterDatasetRequest",
    "GetDatasetRequest",
    "UnregisterDatasetRequest",
    "fastapi_routes",
]
