# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx
import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


@pytest.fixture(scope="session")
def agents_base_url(llama_stack_client):
    """Provide the base URL for the Agents API, skipping library client mode."""
    if isinstance(llama_stack_client, LlamaStackAsLibraryClient):
        pytest.skip("Agents API tests are not supported in library client mode")
    return llama_stack_client.base_url


@pytest.fixture
def agents_client(agents_base_url):
    """Provide an httpx client configured for Anthropic Agents API calls."""
    client = httpx.Client(base_url=f"{agents_base_url}/v1alpha", timeout=30.0)
    yield client
    client.close()
