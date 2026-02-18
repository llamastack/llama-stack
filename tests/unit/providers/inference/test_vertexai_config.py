# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from llama_stack.providers.remote.inference.vertexai.config import (
    VertexAIConfig,
    VertexAIProviderDataValidator,
)


@pytest.fixture
def project_name() -> str:
    return "test-project"


class TestVertexAIConfig:
    """Test VertexAIConfig initialization and field handling."""

    @pytest.mark.parametrize(
        "config_kwargs,expected_location,expected_auth_credential,expected_access_token",
        [
            ({"project": "test-project"}, "global", None, None),
            (
                {
                    "project": "test-project",
                    "location": "us-central1",
                    "auth_credential": SecretStr("service-account-json"),
                    "access_token": SecretStr("test-token"),
                },
                "us-central1",
                "service-account-json",
                "test-token",
            ),
        ],
    )
    def test_config_field_population(
        self,
        config_kwargs,
        expected_location,
        expected_auth_credential,
        expected_access_token,
    ):
        config = VertexAIConfig(**config_kwargs)
        assert config.project == "test-project"
        assert config.location == expected_location
        assert (config.auth_credential.get_secret_value() if config.auth_credential is not None else None) == (
            expected_auth_credential
        )
        assert (config.access_token.get_secret_value() if config.access_token is not None else None) == (
            expected_access_token
        )

    def test_config_explicit_overrides_env(self):
        """Test that explicit values override environment variables."""
        env_vars = {
            "VERTEX_AI_PROJECT": "env-project",
            "VERTEX_AI_LOCATION": "europe-west1",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = VertexAIConfig(
                project="explicit-project",
                location="us-west1",
            )
            assert config.project == "explicit-project"
            assert config.location == "us-west1"

    @pytest.mark.parametrize(
        "sample_kwargs,expected_project,expected_location",
        [
            ({}, "${env.VERTEX_AI_PROJECT:=}", "${env.VERTEX_AI_LOCATION:=global}"),
            (
                {"project": "custom-project", "location": "custom-location"},
                "custom-project",
                "custom-location",
            ),
        ],
    )
    def test_sample_run_config(self, sample_kwargs, expected_project, expected_location):
        sample = VertexAIConfig.sample_run_config(**sample_kwargs)
        assert sample["project"] == expected_project
        assert sample["location"] == expected_location

    def test_config_missing_required_project(self):
        """Test that project field is required."""
        with pytest.raises(ValidationError):
            VertexAIConfig.model_validate({})

    def test_auth_credential_excluded_from_schema(self, project_name):
        """Test that auth_credential is excluded from serialization."""
        config = VertexAIConfig(
            project=project_name,
            auth_credential=SecretStr("secret-creds"),
        )
        dumped = config.model_dump(exclude_unset=False)
        assert "auth_credential" not in dumped or dumped.get("auth_credential") is None


class TestVertexAIProviderDataValidator:
    """Test VertexAIProviderDataValidator initialization."""

    @pytest.mark.parametrize(
        "validator_kwargs,expected_project,expected_location,expected_access_token",
        [
            ({}, None, None, None),
            (
                {
                    "vertex_project": "test-project",
                    "vertex_location": "us-central1",
                    "vertex_access_token": "test-token",
                },
                "test-project",
                "us-central1",
                "test-token",
            ),
            ({"vertex_project": "test-project"}, "test-project", None, None),
        ],
    )
    def test_validator_field_population(
        self,
        validator_kwargs,
        expected_project,
        expected_location,
        expected_access_token,
    ):
        validator = VertexAIProviderDataValidator(**validator_kwargs)
        assert validator.vertex_project == expected_project
        assert validator.vertex_location == expected_location
        assert validator.vertex_access_token == expected_access_token


class TestVertexAIConfigBackwardCompatibility:
    """Test backward compatibility of VertexAIConfig changes."""

    def test_config_with_auth_credential_still_works(self, project_name):
        """Test that auth_credential field still works."""
        config = VertexAIConfig(
            project=project_name,
            auth_credential=SecretStr("creds"),
        )
        assert config.project == project_name
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == "creds"
