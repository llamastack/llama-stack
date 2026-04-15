# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Re-export from canonical location in utils package.
# This shim exists for backward compatibility with existing provider code.
# Uses sys.modules aliasing so that unittest.mock.patch works correctly.
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_stack_utils_inference.http_client import *  # noqa: F401, F403

import llama_stack_utils_inference.http_client as _canonical

sys.modules[__name__] = _canonical
