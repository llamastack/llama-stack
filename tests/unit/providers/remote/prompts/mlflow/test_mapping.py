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
        return PromptIDMapper(use_metadata=True)

    def test_to_mlflow_name_valid_id(self, mapper):
        """Test converting valid prompt_id to MLflow name."""
        prompt_id = "pmpt_" + "a" * 48
        mlflow_name = mapper.to_mlflow_name(prompt_id)

        assert mlflow_name == "llama_prompt_" + "a" * 48
        assert mlflow_name.startswith(mapper.MLFLOW_NAME_PREFIX)

    def test_to_mlflow_name_different_hex(self, mapper):
        """Test conversion with different hex values."""
        prompt_id = "pmpt_0123456789abcdef" + "f" * 32
        mlflow_name = mapper.to_mlflow_name(prompt_id)

        assert mlflow_name == "llama_prompt_0123456789abcdef" + "f" * 32

    def test_to_mlflow_name_invalid_prefix(self, mapper):
        """Test conversion fails with invalid prefix."""
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("invalid_" + "a" * 48)

    def test_to_mlflow_name_wrong_length(self, mapper):
        """Test conversion fails with wrong hex length."""
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "a" * 47)  # Too short

        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "a" * 49)  # Too long

    def test_to_mlflow_name_uppercase_hex(self, mapper):
        """Test conversion fails with uppercase hex."""
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "A" * 48)

    def test_to_mlflow_name_invalid_hex_chars(self, mapper):
        """Test conversion fails with invalid hex characters."""
        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "g" * 48)  # 'g' is not hex

        with pytest.raises(ValueError, match="Invalid prompt_id format"):
            mapper.to_mlflow_name("pmpt_" + "xyz" + "a" * 45)

    def test_to_llama_id_valid_name(self, mapper):
        """Test converting valid MLflow name to prompt_id."""
        mlflow_name = "llama_prompt_" + "b" * 48
        prompt_id = mapper.to_llama_id(mlflow_name)

        assert prompt_id == "pmpt_" + "b" * 48
        assert prompt_id.startswith("pmpt_")

    def test_to_llama_id_different_hex(self, mapper):
        """Test conversion with different hex values."""
        mlflow_name = "llama_prompt_fedcba9876543210" + "0" * 32
        prompt_id = mapper.to_llama_id(mlflow_name)

        assert prompt_id == "pmpt_fedcba9876543210" + "0" * 32

    def test_to_llama_id_invalid_prefix(self, mapper):
        """Test conversion fails with invalid prefix."""
        with pytest.raises(ValueError, match="does not start with expected prefix"):
            mapper.to_llama_id("wrong_prefix_" + "a" * 48)

    def test_to_llama_id_wrong_hex_length(self, mapper):
        """Test conversion fails with wrong hex length."""
        with pytest.raises(ValueError, match="Invalid hex part length"):
            mapper.to_llama_id("llama_prompt_" + "a" * 47)

        with pytest.raises(ValueError, match="Invalid hex part length"):
            mapper.to_llama_id("llama_prompt_" + "a" * 49)

    def test_to_llama_id_invalid_hex_chars(self, mapper):
        """Test conversion fails with invalid hex characters."""
        with pytest.raises(ValueError, match="Invalid character"):
            mapper.to_llama_id("llama_prompt_" + "G" * 48)

        with pytest.raises(ValueError, match="Invalid character"):
            mapper.to_llama_id("llama_prompt_xyz" + "a" * 45)

    def test_bidirectional_conversion(self, mapper):
        """Test bidirectional conversion preserves IDs."""
        original_id = "pmpt_0123456789abcdef" + "a" * 32

        # Convert to MLflow name and back
        mlflow_name = mapper.to_mlflow_name(original_id)
        recovered_id = mapper.to_llama_id(mlflow_name)

        assert recovered_id == original_id

    def test_bidirectional_conversion_reverse(self, mapper):
        """Test bidirectional conversion starting from MLflow name."""
        original_name = "llama_prompt_fedcba9876543210" + "f" * 32

        # Convert to prompt_id and back
        prompt_id = mapper.to_llama_id(original_name)
        recovered_name = mapper.to_mlflow_name(prompt_id)

        assert recovered_name == original_name

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

    def test_get_metadata_tags_empty_variables(self, mapper):
        """Test metadata tags with empty variables list."""
        prompt_id = "pmpt_" + "e" * 48
        variables = []

        tags = mapper.get_metadata_tags(prompt_id, variables)

        assert tags["llama_stack_id"] == prompt_id
        assert tags["llama_stack_managed"] == "true"
        assert "variables" not in tags

    def test_get_metadata_tags_disabled(self):
        """Test metadata tags returns empty dict when disabled."""
        mapper = PromptIDMapper(use_metadata=False)
        prompt_id = "pmpt_" + "f" * 48
        variables = ["var1"]

        tags = mapper.get_metadata_tags(prompt_id, variables)

        assert tags == {}

    def test_extract_variables_from_tags(self, mapper):
        """Test extracting variables from tags."""
        tags = {"variables": "var1,var2,var3"}

        variables = mapper.extract_variables_from_tags(tags)

        assert variables == ["var1", "var2", "var3"]

    def test_extract_variables_from_tags_single(self, mapper):
        """Test extracting single variable from tags."""
        tags = {"variables": "single_var"}

        variables = mapper.extract_variables_from_tags(tags)

        assert variables == ["single_var"]

    def test_extract_variables_from_tags_empty(self, mapper):
        """Test extracting variables from empty tags."""
        tags = {"variables": ""}

        variables = mapper.extract_variables_from_tags(tags)

        assert variables == []

    def test_extract_variables_from_tags_missing(self, mapper):
        """Test extracting variables when key is missing."""
        tags = {}

        variables = mapper.extract_variables_from_tags(tags)

        assert variables == []

    def test_extract_variables_from_tags_with_whitespace(self, mapper):
        """Test extracting variables strips whitespace."""
        tags = {"variables": " var1 , var2 , var3 "}

        variables = mapper.extract_variables_from_tags(tags)

        assert variables == ["var1", "var2", "var3"]

    def test_deterministic_mapping(self, mapper):
        """Test that same ID always maps to same name."""
        prompt_id = "pmpt_" + "abc123" + "0" * 42

        name1 = mapper.to_mlflow_name(prompt_id)
        name2 = mapper.to_mlflow_name(prompt_id)

        assert name1 == name2

    def test_unique_mapping(self, mapper):
        """Test that different IDs map to different names."""
        id1 = "pmpt_" + "a" * 48
        id2 = "pmpt_" + "b" * 48

        name1 = mapper.to_mlflow_name(id1)
        name2 = mapper.to_mlflow_name(id2)

        assert name1 != name2

    def test_mlflow_name_prefix_constant(self, mapper):
        """Test MLFLOW_NAME_PREFIX constant value."""
        assert mapper.MLFLOW_NAME_PREFIX == "llama_prompt_"

    def test_prompt_id_pattern_matches_valid(self, mapper):
        """Test PROMPT_ID_PATTERN matches valid IDs."""
        valid_ids = [
            "pmpt_" + "0" * 48,
            "pmpt_" + "f" * 48,
            "pmpt_0123456789abcdef" + "a" * 32,
        ]

        for prompt_id in valid_ids:
            assert mapper.PROMPT_ID_PATTERN.match(prompt_id)

    def test_prompt_id_pattern_rejects_invalid(self, mapper):
        """Test PROMPT_ID_PATTERN rejects invalid IDs."""
        invalid_ids = [
            "pmpt_" + "A" * 48,  # Uppercase
            "pmpt_" + "g" * 48,  # Invalid hex
            "pmpt_" + "a" * 47,  # Too short
            "pmpt_" + "a" * 49,  # Too long
            "invalid_" + "a" * 48,  # Wrong prefix
        ]

        for prompt_id in invalid_ids:
            assert not mapper.PROMPT_ID_PATTERN.match(prompt_id)
