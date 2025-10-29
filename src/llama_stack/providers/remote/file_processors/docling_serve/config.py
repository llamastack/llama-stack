# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class DoclingServeConfig(BaseModel):
    base_url: str = Field(..., description="Base URL of the Docling Serve endpoint")
    api_key: str | None = Field(default=None, description="API key for authentication")
    timeout_seconds: int = Field(default=120, ge=1, le=600, description="Request timeout in seconds")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")

    @staticmethod
    def sample_run_config(**kwargs):
        return {
            "base_url": "http://localhost:8080",
            "api_key": None,
            "timeout_seconds": 120,
            "max_file_size_mb": 100,
        }