# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from typing import Any

from pydantic import BaseModel, Field


class NVIDIAEvalConfig(BaseModel):
    """
     Configuration for the NVIDIA NeMo Evaluator microservice endpoint.

    Attributes:
        evaluator_url (str): A base url for accessing the NVIDIA evaluation endpoint, e.g. http://localhost:8000.
    """

    evaluator_url: str = Field(
        default_factory=lambda: os.getenv("NVIDIA_EVALUATOR_URL", "http://0.0.0.0:7331"),
        description="The url for accessing the evaluator service",
    )

    @classmethod
    def sample_run_config(cls, **kwargs) -> dict[str, Any]:
        return {
            "evaluator_url": "${env.NVIDIA_EVALUATOR_URL:=http://localhost:7331}",
        }
