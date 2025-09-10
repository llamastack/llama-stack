# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest_socket

# We need to import the fixtures here so that pytest can find them
# but ruff doesn't think they are used and removes the import. "noqa: F401" prevents them from being removed
from .fixtures import cached_disk_dist_registry, disk_dist_registry, sqlite_kvstore  # noqa: F401


def pytest_runtest_setup(item):
    """Setup for each test - check if network access should be allowed."""
    if "allow_network" in item.keywords:
        pytest_socket.enable_socket()
    else:
        # Allowing Unix sockets is necessary for some tests that use local servers and mocks
        pytest_socket.disable_socket(allow_unix_socket=True)
