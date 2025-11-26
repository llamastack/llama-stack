# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""ID mapping utilities for MLflow Prompt Registry provider.

This module handles bidirectional mapping between Llama Stack's prompt_id format
(pmpt_<48-hex-chars>) and MLflow's name-based system (llama_prompt_<hex>).
"""

import re


class PromptIDMapper:
    """Handle bidirectional mapping between Llama Stack IDs and MLflow names.

    Llama Stack uses prompt IDs in format: pmpt_<48-hex-chars>
    MLflow uses string names, so we map to: llama_prompt_<48-hex-chars>

    This ensures:
    - Deterministic mapping (same ID always maps to same name)
    - Reversible (can recover original ID from MLflow name)
    - Unique (different IDs map to different names)
    """

    # Regex pattern for Llama Stack prompt_id validation
    PROMPT_ID_PATTERN = re.compile(r"^pmpt_[0-9a-f]{48}$")

    # Prefix for MLflow prompt names managed by Llama Stack
    MLFLOW_NAME_PREFIX = "llama_prompt_"

    def to_mlflow_name(self, prompt_id: str) -> str:
        """Convert Llama Stack prompt_id to MLflow prompt name.

        Args:
            prompt_id: Llama Stack prompt ID (format: pmpt_<48-hex-chars>)

        Returns:
            MLflow prompt name (format: llama_prompt_<48-hex-chars>)

        Raises:
            ValueError: If prompt_id format is invalid

        Example:
            >>> mapper = PromptIDMapper()
            >>> mapper.to_mlflow_name("pmpt_a1b2c3d4e5f6...")
            "llama_prompt_a1b2c3d4e5f6..."
        """
        if not self.PROMPT_ID_PATTERN.match(prompt_id):
            raise ValueError(f"Invalid prompt_id format: {prompt_id}. Expected format: pmpt_<48-hex-chars>")

        # Extract hex part (after "pmpt_" prefix)
        hex_part = prompt_id.split("pmpt_")[1]

        # Create MLflow name
        return f"{self.MLFLOW_NAME_PREFIX}{hex_part}"

    def to_llama_id(self, mlflow_name: str) -> str:
        """Convert MLflow prompt name to Llama Stack prompt_id.

        Args:
            mlflow_name: MLflow prompt name

        Returns:
            Llama Stack prompt ID (format: pmpt_<48-hex-chars>)

        Raises:
            ValueError: If name doesn't follow expected format

        Example:
            >>> mapper = PromptIDMapper()
            >>> mapper.to_llama_id("llama_prompt_a1b2c3d4e5f6...")
            "pmpt_a1b2c3d4e5f6..."
        """
        if not mlflow_name.startswith(self.MLFLOW_NAME_PREFIX):
            raise ValueError(
                f"MLflow name '{mlflow_name}' does not start with expected prefix '{self.MLFLOW_NAME_PREFIX}'"
            )

        # Extract hex part
        hex_part = mlflow_name[len(self.MLFLOW_NAME_PREFIX) :]

        # Validate hex part length and characters
        if len(hex_part) != 48:
            raise ValueError(f"Invalid hex part length in MLflow name '{mlflow_name}'. Expected 48 characters.")

        for char in hex_part:
            if char not in "0123456789abcdef":
                raise ValueError(
                    f"Invalid character '{char}' in hex part of MLflow name '{mlflow_name}'. "
                    "Expected lowercase hex characters [0-9a-f]."
                )

        return f"pmpt_{hex_part}"

    def get_metadata_tags(self, prompt_id: str, variables: list[str] | None = None) -> dict[str, str]:
        """Generate MLflow tags with Llama Stack metadata.

        Args:
            prompt_id: Llama Stack prompt ID
            variables: List of prompt variables (optional)

        Returns:
            Dictionary of MLflow tags for metadata storage

        Example:
            >>> mapper = PromptIDMapper()
            >>> tags = mapper.get_metadata_tags("pmpt_abc123...", ["var1", "var2"])
            >>> tags
            {"llama_stack_id": "pmpt_abc123...", "llama_stack_managed": "true", "variables": "var1,var2"}
        """
        tags = {
            "llama_stack_id": prompt_id,
            "llama_stack_managed": "true",
        }

        if variables:
            # Store variables as comma-separated string
            tags["variables"] = ",".join(variables)

        return tags
