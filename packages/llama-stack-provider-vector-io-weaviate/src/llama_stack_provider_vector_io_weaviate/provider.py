# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.vector_io,
        adapter_type="weaviate",
        provider_type="remote::weaviate",
        pip_packages=[],
        module="llama_stack_provider_vector_io_weaviate",
        config_class="llama_stack_provider_vector_io_weaviate.config.WeaviateVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="Vector I/O provider using weaviate.",
    )
