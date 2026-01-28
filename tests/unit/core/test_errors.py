# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import httpx

from llama_stack_api.common.errors import (
    ClientListCommand,
    ResourceNotFoundError,
)


class TestClientListCommand:
    def test_basic_command(self):
        cmd = ClientListCommand("files.list")
        assert str(cmd) == "Use 'client.files.list()'."

    def test_with_argument(self):
        cmd = ClientListCommand("connectors.list_tools", "my-connector")
        assert str(cmd) == "Use 'client.connectors.list_tools(\"my-connector\")'."

    def test_with_multiple_arguments(self):
        cmd = ClientListCommand("widgets.find", ["arg1", "arg2"])
        assert str(cmd) == 'Use \'client.widgets.find("arg1", "arg2")\'.'

    def test_with_resource_plural(self):
        cmd = ClientListCommand("batches.list", resource_name_plural="batches")
        assert str(cmd) == "Use 'client.batches.list()' to list available batches."

    def test_empty_arguments_list(self):
        cmd = ClientListCommand("models.list", [])
        assert str(cmd) == "Use 'client.models.list()'."

    def test_none_arguments(self):
        cmd = ClientListCommand("models.list", None)
        assert str(cmd) == "Use 'client.models.list()'."


class TestResourceNotFoundError:
    def test_without_client_list(self):
        err = ResourceNotFoundError("test-id", "Widget")
        assert str(err) == "Widget 'test-id' not found."

    def test_with_client_list_infers_plural(self):
        err = ResourceNotFoundError("test-id", "Model", ClientListCommand("models.list"))
        # ResourceNotFoundError sets resource_name_plural to "{resource_type}s", then __str__ lowercases it
        assert str(err) == "Model 'test-id' not found. Use 'client.models.list()' to list available models."

    def test_with_explicit_plural(self):
        err = ResourceNotFoundError(
            "test-id", "Batch", ClientListCommand("batches.list", resource_name_plural="batches")
        )
        assert str(err) == "Batch 'test-id' not found. Use 'client.batches.list()' to list available batches."

    def test_status_code_is_not_found(self):
        err = ResourceNotFoundError("test-id", "Widget")
        assert err.status_code == httpx.codes.NOT_FOUND
