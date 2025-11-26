# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end integration tests for MLflow prompts provider.

These tests require a running MLflow server. See conftest.py for setup instructions.
"""

import pytest


class TestMLflowPromptsEndToEnd:
    """End-to-end tests for MLflow prompts provider."""

    async def test_create_and_retrieve_prompt(self, mlflow_adapter):
        """Test creating a prompt and retrieving it by ID."""
        # Create prompt with variables
        created = await mlflow_adapter.create_prompt(
            prompt="Summarize the following text in {{ num_sentences }} sentences: {{ text }}",
            variables=["num_sentences", "text"],
        )

        # Verify created prompt
        assert created.prompt_id.startswith("pmpt_")
        assert len(created.prompt_id) == 53  # "pmpt_" + 48 hex chars
        assert created.version == 1
        assert created.is_default is True
        assert set(created.variables) == {"num_sentences", "text"}
        assert "{{ num_sentences }}" in created.prompt
        assert "{{ text }}" in created.prompt

        # Retrieve prompt by ID (should get default version)
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)

        assert retrieved.prompt_id == created.prompt_id
        assert retrieved.prompt == created.prompt
        assert retrieved.version == created.version
        assert set(retrieved.variables) == set(created.variables)
        assert retrieved.is_default is True

        # Retrieve specific version
        retrieved_v1 = await mlflow_adapter.get_prompt(created.prompt_id, version=1)

        assert retrieved_v1.prompt_id == created.prompt_id
        assert retrieved_v1.version == 1

    async def test_update_prompt_creates_new_version(self, mlflow_adapter):
        """Test that updating a prompt creates a new version."""
        # Create initial prompt (version 1)
        v1 = await mlflow_adapter.create_prompt(
            prompt="Original prompt with {{ variable }}",
            variables=["variable"],
        )

        assert v1.version == 1
        assert v1.is_default is True

        # Update prompt (should create version 2)
        v2 = await mlflow_adapter.update_prompt(
            prompt_id=v1.prompt_id,
            prompt="Updated prompt with {{ variable }}",
            version=1,
            variables=["variable"],
            set_as_default=True,
        )

        assert v2.prompt_id == v1.prompt_id
        assert v2.version == 2
        assert v2.is_default is True
        assert "Updated" in v2.prompt

        # Verify both versions exist
        versions_response = await mlflow_adapter.list_prompt_versions(v1.prompt_id)
        versions = versions_response.data

        assert len(versions) >= 2
        assert any(v.version == 1 for v in versions)
        assert any(v.version == 2 for v in versions)

        # Verify version 1 still exists
        v1_retrieved = await mlflow_adapter.get_prompt(v1.prompt_id, version=1)
        assert "Original" in v1_retrieved.prompt
        assert v1_retrieved.is_default is False  # No longer default

        # Verify version 2 is default
        default = await mlflow_adapter.get_prompt(v1.prompt_id)
        assert default.version == 2
        assert "Updated" in default.prompt

    async def test_list_prompts_returns_defaults_only(self, mlflow_adapter):
        """Test that list_prompts returns only default versions."""
        # Create multiple prompts
        p1 = await mlflow_adapter.create_prompt(
            prompt="Prompt 1 with {{ var }}",
            variables=["var"],
        )

        p2 = await mlflow_adapter.create_prompt(
            prompt="Prompt 2 with {{ var }}",
            variables=["var"],
        )

        # Update first prompt (creates version 2)
        await mlflow_adapter.update_prompt(
            prompt_id=p1.prompt_id,
            prompt="Prompt 1 updated with {{ var }}",
            version=1,
            variables=["var"],
            set_as_default=True,
        )

        # List all prompts
        response = await mlflow_adapter.list_prompts()
        prompts = response.data

        # Should contain at least our 2 prompts
        assert len(prompts) >= 2

        # Find our prompts in the list
        p1_in_list = next((p for p in prompts if p.prompt_id == p1.prompt_id), None)
        p2_in_list = next((p for p in prompts if p.prompt_id == p2.prompt_id), None)

        assert p1_in_list is not None
        assert p2_in_list is not None

        # p1 should be version 2 (updated version is default)
        assert p1_in_list.version == 2
        assert p1_in_list.is_default is True

        # p2 should be version 1 (original is still default)
        assert p2_in_list.version == 1
        assert p2_in_list.is_default is True

    async def test_list_prompt_versions(self, mlflow_adapter):
        """Test listing all versions of a specific prompt."""
        # Create prompt
        v1 = await mlflow_adapter.create_prompt(
            prompt="Version 1 {{ var }}",
            variables=["var"],
        )

        # Create multiple versions
        _v2 = await mlflow_adapter.update_prompt(
            prompt_id=v1.prompt_id,
            prompt="Version 2 {{ var }}",
            version=1,
            variables=["var"],
        )

        _v3 = await mlflow_adapter.update_prompt(
            prompt_id=v1.prompt_id,
            prompt="Version 3 {{ var }}",
            version=2,
            variables=["var"],
        )

        # List all versions
        versions_response = await mlflow_adapter.list_prompt_versions(v1.prompt_id)
        versions = versions_response.data

        # Should have 3 versions
        assert len(versions) == 3

        # Verify versions are sorted by version number
        assert versions[0].version == 1
        assert versions[1].version == 2
        assert versions[2].version == 3

        # Verify content
        assert "Version 1" in versions[0].prompt
        assert "Version 2" in versions[1].prompt
        assert "Version 3" in versions[2].prompt

        # Only latest should be default
        assert versions[0].is_default is False
        assert versions[1].is_default is False
        assert versions[2].is_default is True

    async def test_set_default_version(self, mlflow_adapter):
        """Test changing which version is the default."""
        # Create prompt and update it
        v1 = await mlflow_adapter.create_prompt(
            prompt="Version 1 {{ var }}",
            variables=["var"],
        )

        _v2 = await mlflow_adapter.update_prompt(
            prompt_id=v1.prompt_id,
            prompt="Version 2 {{ var }}",
            version=1,
            variables=["var"],
        )

        # At this point, _v2 is default
        default = await mlflow_adapter.get_prompt(v1.prompt_id)
        assert default.version == 2

        # Set v1 as default
        updated = await mlflow_adapter.set_default_version(v1.prompt_id, 1)
        assert updated.version == 1
        assert updated.is_default is True

        # Verify default changed
        default = await mlflow_adapter.get_prompt(v1.prompt_id)
        assert default.version == 1
        assert "Version 1" in default.prompt

    async def test_variable_auto_extraction(self, mlflow_adapter):
        """Test automatic variable extraction from template."""
        # Create prompt without explicitly specifying variables
        created = await mlflow_adapter.create_prompt(
            prompt="Extract {{ entity }} from {{ text }} in {{ format }} format",
        )

        # Should auto-extract all variables
        assert set(created.variables) == {"entity", "text", "format"}

        # Retrieve and verify
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)
        assert set(retrieved.variables) == {"entity", "text", "format"}

    async def test_variable_validation(self, mlflow_adapter):
        """Test that variable validation works correctly."""
        # Should fail: template has undeclared variable
        with pytest.raises(ValueError, match="undeclared variables"):
            await mlflow_adapter.create_prompt(
                prompt="Template with {{ var1 }} and {{ var2 }}",
                variables=["var1"],  # Missing var2
            )

    async def test_prompt_not_found(self, mlflow_adapter):
        """Test error handling when prompt doesn't exist."""
        fake_id = "pmpt_" + "0" * 48

        with pytest.raises(ValueError, match="not found"):
            await mlflow_adapter.get_prompt(fake_id)

    async def test_version_not_found(self, mlflow_adapter):
        """Test error handling when version doesn't exist."""
        # Create prompt (version 1)
        created = await mlflow_adapter.create_prompt(
            prompt="Test {{ var }}",
            variables=["var"],
        )

        # Try to get non-existent version
        with pytest.raises(ValueError, match="not found"):
            await mlflow_adapter.get_prompt(created.prompt_id, version=999)

    async def test_update_wrong_version(self, mlflow_adapter):
        """Test that updating with wrong version fails."""
        # Create prompt (version 1)
        created = await mlflow_adapter.create_prompt(
            prompt="Test {{ var }}",
            variables=["var"],
        )

        # Try to update with wrong version number
        with pytest.raises(ValueError, match="not the latest"):
            await mlflow_adapter.update_prompt(
                prompt_id=created.prompt_id,
                prompt="Updated {{ var }}",
                version=999,  # Wrong version
                variables=["var"],
            )

    async def test_delete_not_supported(self, mlflow_adapter):
        """Test that deletion raises NotImplementedError."""
        # Create prompt
        created = await mlflow_adapter.create_prompt(
            prompt="Test {{ var }}",
            variables=["var"],
        )

        # Try to delete (should fail with NotImplementedError)
        with pytest.raises(NotImplementedError, match="does not support deletion"):
            await mlflow_adapter.delete_prompt(created.prompt_id)

        # Verify prompt still exists
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)
        assert retrieved.prompt_id == created.prompt_id

    async def test_complex_template_with_multiple_variables(self, mlflow_adapter):
        """Test prompt with complex template and multiple variables."""
        template = """You are a {{ role }} assistant specialized in {{ domain }}.

Task: {{ task }}

Context:
{{ context }}

Instructions:
1. {{ instruction1 }}
2. {{ instruction2 }}
3. {{ instruction3 }}

Output format: {{ output_format }}
"""

        # Create with auto-extraction
        created = await mlflow_adapter.create_prompt(prompt=template)

        # Should extract all variables
        expected_vars = {
            "role",
            "domain",
            "task",
            "context",
            "instruction1",
            "instruction2",
            "instruction3",
            "output_format",
        }
        assert set(created.variables) == expected_vars

        # Retrieve and verify template preserved
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)
        assert retrieved.prompt == template

    async def test_empty_template(self, mlflow_adapter):
        """Test handling of empty template."""
        # Create prompt with empty template
        created = await mlflow_adapter.create_prompt(
            prompt="",
            variables=[],
        )

        assert created.prompt == ""
        assert created.variables == []

        # Retrieve and verify
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)
        assert retrieved.prompt == ""

    async def test_template_with_no_variables(self, mlflow_adapter):
        """Test template without any variables."""
        template = "This is a static prompt with no variables."

        created = await mlflow_adapter.create_prompt(prompt=template)

        assert created.prompt == template
        assert created.variables == []

        # Retrieve and verify
        retrieved = await mlflow_adapter.get_prompt(created.prompt_id)
        assert retrieved.prompt == template
        assert retrieved.variables == []
