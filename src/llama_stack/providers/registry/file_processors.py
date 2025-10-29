# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, InlineProviderSpec, RemoteProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        # PyPDF - Default provider for backward compatibility
        InlineProviderSpec(
            api=Api.file_processors,
            provider_type="inline::pypdf",
            pip_packages=["pypdf"],
            module="llama_stack.providers.inline.file_processors.pypdf",
            config_class="llama_stack.providers.inline.file_processors.pypdf.PyPDFConfig",
            description="Simple PDF text extraction using PyPDF library. Default processor for backward compatibility."
        ),
        
        # Docling with JobKit - Advanced inline processing
        InlineProviderSpec(
            api=Api.file_processors, 
            provider_type="inline::docling",
            pip_packages=["docling-jobkit", "docling", "torch", "torchvision"],  # Updated with docling-jobkit
            module="llama_stack.providers.inline.file_processors.docling",
            config_class="llama_stack.providers.inline.file_processors.docling.DoclingConfig",
            description="Advanced document processing using Docling JobKit ConvertManager with table/figure extraction"
        ),
        
        # Docling Serve - Remote processing
        RemoteProviderSpec(
            api=Api.file_processors,
            adapter_type="docling_serve",
            provider_type="remote::docling_serve",
            pip_packages=["aiohttp"],
            module="llama_stack.providers.remote.file_processors.docling_serve",
            config_class="llama_stack.providers.remote.file_processors.docling_serve.DoclingServeConfig",
            description="Remote Docling processing via Docling Serve API endpoint"
        ),
    ]