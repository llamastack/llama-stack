# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient

from llama_stack.core.library_client import LlamaStackAsLibraryClient


class TestProviders:
    def test_providers(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        provider_list = llama_stack_client.providers.list()
        assert provider_list is not None
        assert len(provider_list) > 0

        for provider in provider_list:
            pid = provider.provider_id
            provider = llama_stack_client.providers.retrieve(pid)
            assert provider is not None
