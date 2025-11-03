# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .files_service import FileService
from .models import (
    ExpiresAfter,
    ListOpenAIFileResponse,
    OpenAIFileDeleteResponse,
    OpenAIFileObject,
    OpenAIFilePurpose,
)

# Backward compatibility - export Files as alias for FileService
Files = FileService

__all__ = [
    "Files",
    "FileService",
    "OpenAIFileObject",
    "OpenAIFilePurpose",
    "ExpiresAfter",
    "ListOpenAIFileResponse",
    "OpenAIFileDeleteResponse",
]
