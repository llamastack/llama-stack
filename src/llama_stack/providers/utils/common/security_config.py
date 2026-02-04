# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import Field

DEFAULT_TRUSTED_MODEL_PREFIXES = [
    "nomic-ai/",
]


class TrustedModelConfig:
    """
    Mixin for provider configs that need trusted model configuration.

    Controls whether models can execute custom code (trust_remote_code=True).
    Only models from trusted prefixes will be allowed to run custom code.
    """

    trusted_model_prefixes: list[str] = Field(
        default=DEFAULT_TRUSTED_MODEL_PREFIXES,
        description="List of trusted model prefixes/organizations. Models from these sources will be loaded with trust_remote_code=True. Others will be loaded with trust_remote_code=False for security.",
    )

    def is_trusted_model(self, model: str) -> bool:
        """Check if model is from a trusted source based on configured allowlist"""
        for prefix in self.trusted_model_prefixes:
            if model.startswith(prefix):
                return True
        return False
