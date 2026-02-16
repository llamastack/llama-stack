# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.providers.remote.inference.vertexai_native.config import (
    VertexAINativeConfig,
    VertexAINativeProviderDataValidator,
)


@pytest.fixture
def config_factory():
    def _build(**overrides):
        return VertexAINativeConfig(project="test-project", **overrides)

    return _build


def test_config_defaults(config_factory):
    config = config_factory()

    assert config.project == "test-project"
    assert config.location == "global"
    assert config.allowed_models is None
    assert config.refresh_models is False
    assert config.auth_credential is None
    assert config.network is None


@pytest.mark.parametrize(
    ("project", "location"),
    [
        ("my-project", "us-east1"),
        ("my-project", "europe-west4"),
    ],
)
def test_config_field_overrides(project, location):
    config = VertexAINativeConfig(project=project, location=location)

    assert config.project == project
    assert config.location == location


def test_sample_run_config_uses_env_placeholders():
    assert VertexAINativeConfig.sample_run_config() == {
        "project": "${env.VERTEX_AI_PROJECT:=}",
        "location": "${env.VERTEX_AI_LOCATION:=global}",
    }


@pytest.mark.parametrize(
    ("provider_data", "expected_project", "expected_location"),
    [
        ({"vertex_project": "proj-a", "vertex_location": "us-east1"}, "proj-a", "us-east1"),
        ({"vertex_project": "proj-b"}, "proj-b", None),
        ({"vertex_location": "us-west1"}, None, "us-west1"),
    ],
)
def test_provider_data_validator_accepts_vertex_fields(provider_data, expected_project, expected_location):
    validator = VertexAINativeProviderDataValidator(**provider_data)

    assert validator.vertex_project == expected_project
    assert validator.vertex_location == expected_location
