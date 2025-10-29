# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import DoclingConfig
from .docling import DoclingFileProcessorImpl

__all__ = ["DoclingConfig", "DoclingFileProcessorImpl"]


async def get_adapter_impl(config: DoclingConfig, _deps):
    return DoclingFileProcessorImpl(config)