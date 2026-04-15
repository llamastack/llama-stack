# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec, ProviderSpec


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.batches,
        provider_type="inline::reference",
        pip_packages=[],
        module="llama_stack_provider_batches_reference",
        config_class="llama_stack_provider_batches_reference.config.ReferenceBatchesImplConfig",
        api_dependencies=[Api.inference, Api.files, Api.models],
        description="Reference implementation of batches API with KVStore persistence.",
    )
