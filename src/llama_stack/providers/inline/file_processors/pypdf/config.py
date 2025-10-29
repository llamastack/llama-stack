# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class PyPDFConfig(BaseModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Processing timeout in seconds")
    max_file_size_mb: int = Field(default=50, ge=1, le=500, description="Maximum file size in MB")

    @staticmethod
    def sample_run_config(**kwargs):
        return {
            "timeout_seconds": 30,
            "max_file_size_mb": 50,
        }