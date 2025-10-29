# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field


class DoclingConfig(BaseModel):
    timeout_seconds: int = Field(default=120, ge=1, le=600, description="Processing timeout in seconds")
    max_file_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum file size in MB")
    model_cache_dir: str | None = Field(default=None, description="Directory to cache Docling models")
    enable_gpu: bool = Field(default=False, description="Enable GPU acceleration if available")

    @staticmethod
    def sample_run_config(**kwargs):
        return {
            "timeout_seconds": 120,
            "max_file_size_mb": 100,
            "model_cache_dir": None,
            "enable_gpu": False,
        }