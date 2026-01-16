# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.remote.vector_io.oci.config import OCI26aiVectorIOConfig
from llama_stack_api import Api, ProviderSpec


async def get_adapter_impl(config: OCI26aiVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from typing import cast

    from llama_stack.providers.remote.vector_io.oci.oci26ai import OCI26aiVectorIOAdapter
    from llama_stack_api import Files, Inference

    assert isinstance(config, OCI26aiVectorIOConfig), f"Unexpected config type: {type(config)}"
    inference_api = cast(Inference, deps[Api.inference])
    files_api = cast(Files | None, deps.get(Api.files))
    impl = OCI26aiVectorIOAdapter(config, inference_api, files_api)
    await impl.initialize()
    return impl
