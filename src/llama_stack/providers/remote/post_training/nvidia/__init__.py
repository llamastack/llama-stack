# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .config import NvidiaPostTrainingConfig


async def get_adapter_impl(
    config: NvidiaPostTrainingConfig,
    _deps,
):
    from .post_training import NvidiaPostTrainingAdapter

    if not isinstance(config, NvidiaPostTrainingConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")

    impl = NvidiaPostTrainingAdapter(config)
    return impl


__all__ = ["get_adapter_impl", "NvidiaPostTrainingAdapter"]
