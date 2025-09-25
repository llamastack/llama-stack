# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic.types import SecretStr


class MySecretStr(SecretStr):
    """A SecretStr that can accept None values to avoid mypy type errors.

    This is useful for optional secret fields where you want to avoid
    explicit None checks in consuming code.

    We chose to not use the SecretStr from pydantic because it does not allow None values and will
    let the provider's library fail if the secret is not provided.
    """

    def __init__(self, secret_value: str | None = None) -> None:
        SecretStr.__init__(self, secret_value)  # type: ignore[arg-type]
