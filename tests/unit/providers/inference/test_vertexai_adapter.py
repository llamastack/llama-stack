# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import datetime
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import DefaultCredentialsError, GoogleAuthError, RefreshError, TransportError

from llama_stack.providers.remote.inference.vertexai.config import VertexAIConfig
from llama_stack.providers.remote.inference.vertexai.vertexai import (
    TOKEN_EXPIRY_BUFFER_SECONDS,
    VertexAIInferenceAdapter,
)


@pytest.fixture
def vertexai_adapter():
    config = VertexAIConfig(project="test-project", location="global")
    return VertexAIInferenceAdapter(config=config)


@pytest.fixture
def cached_credentials(vertexai_adapter):
    """Adapter with cached credentials from an initial get_api_key() call."""
    with (
        patch("llama_stack.providers.remote.inference.vertexai.vertexai.default") as mock_default,
        patch("llama_stack.providers.remote.inference.vertexai.vertexai.google.auth.transport.requests.Request"),
    ):
        mock_credentials = MagicMock()
        mock_credentials.token = "first-token"
        mock_default.return_value = (mock_credentials, "test-project")

        vertexai_adapter.get_api_key()

        yield mock_credentials, mock_default


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


@pytest.mark.parametrize(
    "location,expected_url",
    [
        pytest.param(
            "global",
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi",
            id="global",
        ),
        pytest.param(
            "",
            "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/endpoints/openapi",
            id="empty-falls-through-to-global",
        ),
        pytest.param(
            "us-central1",
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/endpoints/openapi",
            id="regional",
        ),
    ],
)
def test_get_base_url(location, expected_url):
    config = VertexAIConfig(project="my-project", location=location)
    adapter = VertexAIInferenceAdapter(config=config)
    assert adapter.get_base_url() == expected_url


@pytest.mark.parametrize(
    "valid,expiry_offset,expected",
    [
        pytest.param(None, None, False, id="no-credentials"),
        pytest.param(False, None, False, id="invalid"),
        pytest.param(True, None, False, id="no-expiry"),
        pytest.param(True, TOKEN_EXPIRY_BUFFER_SECONDS - 1, False, id="expiring-soon"),
        pytest.param(True, TOKEN_EXPIRY_BUFFER_SECONDS + 600, True, id="well-ahead"),
    ],
)
def test_is_token_fresh(vertexai_adapter, valid, expiry_offset, expected):
    if valid is not None:
        expiry = None
        if expiry_offset is not None:
            expiry = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(
                seconds=expiry_offset
            )
        vertexai_adapter._credentials = MagicMock(valid=valid, expiry=expiry)
    assert vertexai_adapter._is_token_fresh() is expected


def test_cached_token_reused_when_fresh(vertexai_adapter, cached_credentials):
    mock_credentials, mock_default = cached_credentials
    mock_credentials.valid = True
    mock_credentials.expiry = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(hours=1)
    mock_credentials.token = "cached-token"

    token = vertexai_adapter.get_api_key()

    assert token == "cached-token"
    mock_default.assert_called_once()
    mock_credentials.refresh.assert_called_once()


def test_stale_token_triggers_refresh(vertexai_adapter, cached_credentials):
    mock_credentials, mock_default = cached_credentials
    mock_credentials.valid = False
    mock_credentials.token = "refreshed-token"

    token = vertexai_adapter.get_api_key()

    assert token == "refreshed-token"
    mock_default.assert_called_once()
    assert mock_credentials.refresh.call_count == 2
