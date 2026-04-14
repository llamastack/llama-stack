# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.inference,
        adapter_type="oci",
        provider_type="remote::oci",
        pip_packages=[],
        module="llama_stack_provider_inference_oci",
        config_class="llama_stack_provider_inference_oci.config.OCIConfig",
        provider_data_validator="llama_stack_provider_inference_oci.config.OCIProviderDataValidator",
        description="Oracle Cloud Infrastructure (OCI) Generative AI inference provider.",
    )
