# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .datasetio_service import DatasetIOService, DatasetStore

# Backward compatibility - export DatasetIO as alias for DatasetIOService
DatasetIO = DatasetIOService

__all__ = ["DatasetIO", "DatasetIOService", "DatasetStore"]
