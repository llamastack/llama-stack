# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, ConfigDict, SecretStr

from .config import PassthroughImplConfig


class PassthroughProviderDataValidator(BaseModel):
    # Lives here because the framework resolves provider_data_validator by module path,
    # and the registry entry points to this package root.
    #
    # extra="allow" because forward_headers key names (e.g. "maas_api_token") are
    # deployer-defined at config time — they can't be declared as typed fields.
    # Without it, Pydantic drops them before build_forwarded_headers() can read them.
    model_config = ConfigDict(extra="allow")

    passthrough_url: str | None = None
    passthrough_api_key: SecretStr | None = None


async def get_adapter_impl(config: PassthroughImplConfig, _deps):
    from .passthrough import PassthroughInferenceAdapter

    if not isinstance(config, PassthroughImplConfig):
        raise ValueError(f"Unexpected config type: {type(config)}")
    impl = PassthroughInferenceAdapter(config)
    await impl.initialize()
    return impl
