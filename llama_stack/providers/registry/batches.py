# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.batches,
            provider_type="inline::reference",
            pip_packages=[],
            module="llama_stack.providers.inline.batches.reference",
            config_class="llama_stack.providers.inline.batches.reference.config.ReferenceBatchesImplConfig",
            api_dependencies=[
                Api.inference,
                Api.files,
                Api.models,
            ],
            description="Reference implementation of batches API with KVStore persistence.",
        ),
    ]
