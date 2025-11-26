# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import NvidiaDatasetIOConfig


async def get_adapter_impl(
    config: NvidiaDatasetIOConfig,
    _deps,
):
    from .datasetio import NvidiaDatasetIOAdapter

    if not isinstance(config, NvidiaDatasetIOConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")

    impl = NvidiaDatasetIOAdapter(config)
    return impl


__all__ = ["get_adapter_impl", "NvidiaDatasetIOAdapter"]
