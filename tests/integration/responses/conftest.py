# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.core.library_client import LlamaStackAsLibraryClient


@pytest.fixture
def responses_client(compat_client):
    """Provide a client for responses tests, skipping library client mode."""
    if isinstance(compat_client, LlamaStackAsLibraryClient):
        pytest.skip("Responses API tests are not supported in library client mode")
    return compat_client
