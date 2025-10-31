# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import InferenceAPIImplConfig, InferenceEndpointImplConfig, TGIImplConfig


async def get_adapter_impl(
    config: InferenceAPIImplConfig | InferenceEndpointImplConfig | TGIImplConfig,
    _deps,
):
    from .tgi import InferenceAPIAdapter, InferenceEndpointAdapter, TGIAdapter

    if isinstance(config, TGIImplConfig):
        impl = TGIAdapter()
    elif isinstance(config, InferenceAPIImplConfig):
        impl = InferenceAPIAdapter()
    elif isinstance(config, InferenceEndpointImplConfig):
        impl = InferenceEndpointAdapter()
    else:
        raise ValueError(
            f"Invalid configuration. Expected 'TGIAdapter', 'InferenceAPIImplConfig' or 'InferenceEndpointImplConfig'. Got {type(config)}."
        )

    await impl.initialize(config)
    return impl
