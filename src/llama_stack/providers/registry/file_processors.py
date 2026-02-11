# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf"],
            module="llama_stack.providers.inline.file_processor.pypdf",
            config_class="llama_stack.providers.inline.file_processor.pypdf.PyPDFFileProcessorConfig",
            api_dependencies=[Api.files],
            description="PyPDF-based file processor for extracting text content from documents.",
        ),
    ]
