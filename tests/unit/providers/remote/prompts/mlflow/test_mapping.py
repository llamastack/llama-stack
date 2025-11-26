# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for MLflow prompts ID mapping utilities."""

import pytest

from llama_stack.providers.remote.prompts.mlflow.mapping import PromptIDMapper


class TestPromptIDMapper:
    """Tests for PromptIDMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create ID mapper instance."""
        return PromptIDMapper()

    def test_to_mlflow_name_valid(self, mapper):
        """Test converting valid prompt_id to MLflow name."""
        prompt_id = "pmpt_" + "a" * 48
        mlflow_name = mapper.to_mlflow_name(prompt_id)

        assert mlflow_name == "llama_prompt_" + "a" * 48
        assert mlflow_name.startswith(mapper.MLFLOW_NAME_PREFIX)

    def test_to_mlflow_name_invalid(self, mapper):
        """Test conversion fails with invalid inputs."""
        # Invalid prefix
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("invalid_" + "a" * 48)

        # Wrong length
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "a" * 47)

        # Invalid hex characters
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "g" * 48)

    def test_to_llama_id_valid(self, mapper):
        """Test converting valid MLflow name to prompt_id."""
        mlflow_name = "llama_prompt_" + "b" * 48
        prompt_id = mapper.to_llama_id(mlflow_name)

        assert prompt_id == "pmpt_" + "b" * 48
        assert prompt_id.startswith("pmpt_")

    def test_to_llama_id_invalid(self, mapper):
        """Test conversion fails with invalid inputs."""
        # Invalid prefix
        with pytest.raises(ValueError, match="does not start with expected prefix"):
            mapper.to_llama_id("wrong_prefix_" + "a" * 48)

        # Wrong length
        with pytest.raises(ValueError, match="Invalid hex part length"):
            mapper.to_llama_id("llama_prompt_" + "a" * 47)

        # Invalid hex characters
        with pytest.raises(ValueError, match="Invalid character"):
            mapper.to_llama_id("llama_prompt_" + "G" * 48)

    def test_bidirectional_conversion(self, mapper):
        """Test bidirectional conversion preserves IDs."""
        original_id = "pmpt_0123456789abcdef" + "a" * 32

        # Convert to MLflow name and back
        mlflow_name = mapper.to_mlflow_name(original_id)
        recovered_id = mapper.to_llama_id(mlflow_name)

        assert recovered_id == original_id

    def test_get_metadata_tags_with_variables(self, mapper):
        """Test metadata tags generation with variables."""
        prompt_id = "pmpt_" + "c" * 48
        variables = ["var1", "var2", "var3"]

        tags = mapper.get_metadata_tags(prompt_id, variables)

        assert tags["llama_stack_id"] == prompt_id
        assert tags["llama_stack_managed"] == "true"
        assert tags["variables"] == "var1,var2,var3"

    def test_get_metadata_tags_without_variables(self, mapper):
        """Test metadata tags generation without variables."""
        prompt_id = "pmpt_" + "d" * 48

        tags = mapper.get_metadata_tags(prompt_id)

        assert tags["llama_stack_id"] == prompt_id
        assert tags["llama_stack_managed"] == "true"
        assert "variables" not in tags
