# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import warnings


def pytest_sessionstart(session) -> None:
    if "LLAMA_STACK_LOGGING" not in os.environ:
        os.environ["LLAMA_STACK_LOGGING"] = "all=WARNING"

    # Silence common deprecation spam during unit tests.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


pytest_plugins = ["tests.unit.fixtures"]
