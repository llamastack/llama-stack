# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import PyPDFConfig
from .pypdf import PyPDFFileProcessorImpl

__all__ = ["PyPDFConfig", "PyPDFFileProcessorImpl"]


async def get_adapter_impl(config: PyPDFConfig, _deps):
    return PyPDFFileProcessorImpl(config)