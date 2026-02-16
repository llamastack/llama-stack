# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, RefreshError, TransportError

from llama_stack.providers.remote.inference.vertexai.config import VertexAIConfig
from llama_stack.providers.remote.inference.vertexai.vertexai import VertexAIInferenceAdapter


@pytest.fixture
def vertexai_adapter():
    config = VertexAIConfig(project="test-project", location="global")
    return VertexAIInferenceAdapter(config=config)


@patch("llama_stack.providers.remote.inference.vertexai.vertexai.google.auth.transport.requests.Request")
@patch("llama_stack.providers.remote.inference.vertexai.vertexai.default")
def test_get_api_key_success(mock_default, mock_request, vertexai_adapter):
    """ADC happy path: credentials refresh and return a valid token."""
    mock_credentials = MagicMock()
    mock_credentials.token = "test-access-token"
    mock_default.return_value = (mock_credentials, "test-project")

    token = vertexai_adapter.get_api_key()

    assert token == "test-access-token"
    mock_credentials.refresh.assert_called_once_with(mock_request.return_value)


@pytest.mark.parametrize(
    "exception_cls,raise_on,expected_message",
    [
        (DefaultCredentialsError, "default", "No credentials found"),
        (RefreshError, "refresh", "Token refresh failed"),
        (TransportError, "refresh", "Network connectivity"),
        (GoogleAuthError, "refresh", "authentication failed"),
    ],
    ids=["no-credentials", "refresh-failure", "network-error", "generic-auth-error"],
)
@patch("llama_stack.providers.remote.inference.vertexai.vertexai.default")
def test_get_api_key_auth_errors(mock_default, vertexai_adapter, exception_cls, raise_on, expected_message):
    """ADC error paths raise ValueError with actionable messages and chained cause."""
    original_error = exception_cls("original error")

    if raise_on == "default":
        mock_default.side_effect = original_error
    else:
        mock_credentials = MagicMock()
        mock_credentials.refresh.side_effect = original_error
        mock_default.return_value = (mock_credentials, "test-project")

    with pytest.raises(ValueError, match=expected_message) as exc_info:
        vertexai_adapter.get_api_key()

    assert exc_info.value.__cause__ is original_error


def test_get_base_url_global():
    """Global location uses the non-regional endpoint."""
    config = VertexAIConfig(project="my-project", location="global")
    adapter = VertexAIInferenceAdapter(config=config)

    assert adapter.get_base_url() == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi"
    )


def test_get_base_url_empty_location():
    """Empty location string falls through to the global endpoint."""
    config = VertexAIConfig(project="my-project", location="")
    adapter = VertexAIInferenceAdapter(config=config)

    assert adapter.get_base_url() == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi"
    )


def test_get_base_url_regional():
    """Regional location uses the location-prefixed endpoint."""
    config = VertexAIConfig(project="my-project", location="us-central1")
    adapter = VertexAIInferenceAdapter(config=config)

    assert adapter.get_base_url() == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/openapi"
    )
