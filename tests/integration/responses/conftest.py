# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient

# Import fixtures from common module to make them available in this test directory
from tests.integration.fixtures.common import (  # noqa: F401
    openai_client,
    require_server,
)


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation.

    When tests intentionally trigger errors, the server writes extensive error logs to stderr.
    Since stderr is piped to a subprocess buffer with limited pre-allocated memory, the buffer
    can fill up and cause the server's logging calls to block, making the server unresponsive.
    Most tests don't need to inspect server logs, so disabling the pipe prevents this deadlock.

    This runs before any fixtures, ensuring the server starts with stderr disabled.
    """
    os.environ["LLAMA_STACK_TEST_LOG_STDERR"] = "0"


@pytest.fixture
def responses_client(compat_client):
    """Provide a client for responses tests, skipping library client mode."""
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API tests are not supported in library client mode")
    return compat_client
