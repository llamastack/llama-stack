# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.inference import InferenceProvider

from .config import OCIConfig


async def get_adapter_impl(config: OCIConfig, _deps) -> InferenceProvider:
    from .oci import OCIInferenceAdapter

    adapter = OCIInferenceAdapter(config=config)
    await adapter.initialize()
    return adapter
