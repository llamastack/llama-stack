# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .datasets_service import DatasetsService
from .models import (
    CommonDatasetFields,
    Dataset,
    DatasetInput,
    DatasetPurpose,
    DatasetType,
    DataSource,
    ListDatasetsResponse,
    RegisterDatasetRequest,
    RowsDataSource,
    URIDataSource,
)

# Backward compatibility - export Datasets as alias for DatasetsService
Datasets = DatasetsService

__all__ = [
    "Datasets",
    "DatasetsService",
    "Dataset",
    "DatasetInput",
    "CommonDatasetFields",
    "DatasetPurpose",
    "DatasetType",
    "DataSource",
    "URIDataSource",
    "RowsDataSource",
    "ListDatasetsResponse",
    "RegisterDatasetRequest",
]
