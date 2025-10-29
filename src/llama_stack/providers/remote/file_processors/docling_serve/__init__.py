# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DoclingServeConfig
from .docling_serve import DoclingServeFileProcessorImpl

__all__ = ["DoclingServeConfig", "DoclingServeFileProcessorImpl"]


async def get_adapter_impl(config: DoclingServeConfig, _deps):
    return DoclingServeFileProcessorImpl(config)