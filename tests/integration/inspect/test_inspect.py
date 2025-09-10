# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient

from llama_stack import LlamaStackAsLibraryClient


class TestInspect:
    def test_health(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        health = llama_stack_client.inspect.health()
        assert health is not None
        assert health.status == "OK"

    def test_version(self, llama_stack_client: LlamaStackAsLibraryClient | LlamaStackClient):
        version = llama_stack_client.inspect.version()
        assert version is not None
        assert version.version is not None
