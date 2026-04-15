# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Re-export from canonical location in utils package.
# This shim exists for backward compatibility with existing provider code.
from llama_stack_utils_vector_io.vector_utils import *  # noqa: F401, F403
