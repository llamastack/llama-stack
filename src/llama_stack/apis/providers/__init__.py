# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Import routes to trigger router registration
from . import routes  # noqa: F401
from .models import ListProvidersResponse, ProviderInfo
from .providers_service import ProviderService

# Backward compatibility - export Providers as alias for ProviderService
Providers = ProviderService

__all__ = ["Providers", "ProviderService", "ListProvidersResponse", "ProviderInfo"]
