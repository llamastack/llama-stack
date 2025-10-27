# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from typing import Protocol


class KVStore(Protocol):
    # TODO: make the value type bytes instead of str
    async def set(self, key: str, value: str, expiration: datetime | None = None) -> None: ...

    async def get(self, key: str) -> str | None: ...

    async def delete(self, key: str) -> None: ...

    async def values_in_range(self, start_key: str, end_key: str) -> list[str]: ...

    async def keys_in_range(self, start_key: str, end_key: str) -> list[str]: ...
