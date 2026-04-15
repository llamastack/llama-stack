# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::docling",
        pip_packages=[],
        module="llama_stack_provider_file_processor_docling",
        config_class="llama_stack_provider_file_processor_docling.config.DoclingFileProcessorConfig",
        api_dependencies=[Api.files],
        description="Docling file processor for layout-aware, structure-preserving document parsing.",
    )
