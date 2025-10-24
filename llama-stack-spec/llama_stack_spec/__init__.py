# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Llama Stack Specification Package

This package contains API protocol definitions and provider specifications for Llama Stack.
It is a lightweight package with minimal dependencies, designed for:
- Third-party provider developers who need the specs
- Client libraries that need type definitions
- Documentation generation

This package does NOT contain:
- Server implementation (see llama-stack)
- Provider implementations (see llama-stack)
- CLI tools (see llama-stack)
"""

__version__ = "0.1.0"

# Re-export main components
from . import (
    apis,  # noqa: F401
    providers,  # noqa: F401
    strong_typing,  # noqa: F401
)
