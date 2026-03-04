# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack_api import Api

from .config import DoclingFileProcessorConfig


async def get_provider_impl(config: DoclingFileProcessorConfig, deps: dict[Api, Any]):
    """Get the Docling file processor implementation."""
    from .adapter import DoclingFileProcessorAdapter

    assert isinstance(config, DoclingFileProcessorConfig), f"Unexpected config type: {type(config)}"

    files_api = deps.get(Api.files)

    impl = DoclingFileProcessorAdapter(config, files_api)
    return impl


__all__ = ["DoclingFileProcessorConfig", "get_provider_impl"]
