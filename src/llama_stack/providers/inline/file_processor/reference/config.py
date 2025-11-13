# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel


class ReferenceFileProcessorImplConfig(BaseModel):
    """Configuration for the reference file processor implementation."""

    @staticmethod
    def sample_run_config(**kwargs):
        return {}
