# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Re-export from canonical location in core.
# This shim exists for backward compatibility with existing provider code.
from llama_stack.core.routers.inference_store import InferenceStore

__all__ = ["InferenceStore"]
