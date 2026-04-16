# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.vector_io,
        provider_type="inline::milvus",
        pip_packages=[],
        module="llama_stack_provider_vector_io_milvus_inline",
        config_class="llama_stack_provider_vector_io_milvus.inline_config.MilvusVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="""
Please refer to the remote provider documentation.
""",
    )
