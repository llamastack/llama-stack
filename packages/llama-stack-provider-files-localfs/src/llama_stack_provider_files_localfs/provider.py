# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_api import Api, InlineProviderSpec


def get_provider_spec() -> InlineProviderSpec:
    return InlineProviderSpec(
        api=Api.files,
        provider_type="inline::localfs",
        pip_packages=[],
        module="llama_stack_provider_files_localfs",
        config_class="llama_stack_provider_files_localfs.config.LocalfsFilesImplConfig",
        description="Local filesystem-based file storage provider for managing files and documents locally.",
    )
