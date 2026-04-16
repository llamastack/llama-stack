# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, RemoteProviderSpec


def get_provider_spec() -> RemoteProviderSpec:
    return RemoteProviderSpec(
        api=Api.files,
        provider_type="remote::s3",
        adapter_type="s3",
        pip_packages=[],
        module="llama_stack_provider_files_s3",
        config_class="llama_stack_provider_files_s3.config.S3FilesImplConfig",
        description="AWS S3-based file storage provider for scalable cloud file management with metadata persistence.",
    )
