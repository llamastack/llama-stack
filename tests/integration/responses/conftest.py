# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os


def pytest_configure(config):
    """Disable stderr pipe to prevent Rich logging from blocking on buffer saturation.

    This runs before any fixtures, ensuring the server starts with stderr disabled.
    """
    os.environ["LLAMA_STACK_TEST_LOG_STDERR"] = "0"
