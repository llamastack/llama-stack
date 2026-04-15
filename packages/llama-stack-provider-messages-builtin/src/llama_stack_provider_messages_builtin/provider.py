# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec, ProviderSpec


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.messages,
        provider_type="inline::builtin",
        pip_packages=[],
        module="llama_stack_provider_messages_builtin",
        config_class="llama_stack_provider_messages_builtin.config.MessagesConfig",
        api_dependencies=[Api.inference],
        description="Implements the Anthropic Messages API with native passthrough and automatic translation modes.",
    )
