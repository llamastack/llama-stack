# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.file_processors,
        provider_type="inline::pypdf",
        pip_packages=[],
        module="llama_stack_provider_file_processor_pypdf",
        config_class="llama_stack_provider_file_processor_pypdf.config.PyPDFFileProcessorConfig",
        api_dependencies=[Api.files],
        description="PyPDF-based file processor for extracting text content from documents.",
    )
