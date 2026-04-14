# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.vector_io,
        provider_type="inline::faiss",
        pip_packages=[],
        module="llama_stack_provider_vector_io_faiss",
        config_class="llama_stack_provider_vector_io_faiss.config.FaissVectorIOConfig",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="Vector I/O provider using faiss.",
    )


def get_deprecated_builtin_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.vector_io,
        provider_type="inline::builtin",
        pip_packages=[],
        module="llama_stack_provider_vector_io_faiss",
        config_class="llama_stack_provider_vector_io_faiss.config.FaissVectorIOConfig",
        deprecation_warning="Please use the `inline::faiss` provider instead.",
        api_dependencies=[Api.inference],
        optional_api_dependencies=[Api.files, Api.models, Api.file_processors],
        description="Deprecated alias for inline::faiss.",
    )
