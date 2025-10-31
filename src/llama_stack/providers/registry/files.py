# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec, RemoteProviderSpec
from llama_stack.providers.utils.sqlstore.sqlstore import sql_store_pip_packages


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.files,
            provider_type="inline::localfs",
            # TODO: make this dynamic according to the sql store type
            pip_packages=sql_store_pip_packages,
            module="llama_stack.providers.inline.files.localfs",
            config_class="llama_stack.providers.inline.files.localfs.config.LocalfsFilesImplConfig",
            description="Local filesystem-based file storage provider for managing files and documents locally.",
        ),
        RemoteProviderSpec(
            api=Api.files,
            provider_type="remote::s3",
            adapter_type="s3",
            pip_packages=["boto3"] + sql_store_pip_packages,
            module="llama_stack.providers.remote.files.s3",
            config_class="llama_stack.providers.remote.files.s3.config.S3FilesImplConfig",
            description="AWS S3-based file storage provider for scalable cloud file management with metadata persistence.",
        ),
        RemoteProviderSpec(
            api=Api.files,
            provider_type="remote::openai",
            adapter_type="openai",
            pip_packages=["openai"] + sql_store_pip_packages,
            module="llama_stack.providers.remote.files.openai",
            config_class="llama_stack.providers.remote.files.openai.config.OpenAIFilesImplConfig",
            description="OpenAI Files API provider for managing files through OpenAI's native file storage service.",
        ),
    ]
