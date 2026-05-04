# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from ogx_api import (
    Api,
    ProviderSpec,
    RemoteProviderSpec,
)


def available_providers() -> list[ProviderSpec]:
    """Return the list of available safety provider specifications.

    Returns:
        List of ProviderSpec objects describing available providers
    """
    return [
        RemoteProviderSpec(
            api=Api.safety,
            adapter_type="passthrough",
            provider_type="remote::passthrough",
            pip_packages=[],
            module="ogx.providers.remote.safety.passthrough",
            config_class="ogx.providers.remote.safety.passthrough.PassthroughSafetyConfig",
            provider_data_validator="ogx.providers.remote.safety.passthrough.config.PassthroughProviderDataValidator",
            description="Passthrough safety provider that forwards moderation calls to a downstream HTTP service.",
        ),
    ]
