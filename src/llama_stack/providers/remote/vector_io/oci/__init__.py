# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, ProviderSpec

from llama_stack.providers.remote.vector_io.oci.config import OCI26aiVectorIOConfig


async def get_adapter_impl(config: OCI26aiVectorIOConfig, deps: dict[Api, ProviderSpec]):
    from llama_stack.providers.remote.vector_io.oci.oci26ai import OCI26aiVectorIOAdapter

    assert isinstance(config, OCI26aiVectorIOConfig), f"Unexpected config type: {type(config)}"
    impl = OCI26aiVectorIOAdapter(config, deps[Api.inference], deps.get(Api.files))
    await impl.initialize()
    return impl
