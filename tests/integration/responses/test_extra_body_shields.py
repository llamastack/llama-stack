# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Test for extra_body parameter support with shields example.

This test demonstrates that parameters marked with ExtraBodyField annotation
can be passed via extra_body in the client SDK and are received by the
server-side implementation.
"""

import pytest
from llama_stack_client import APIStatusError


def test_shields_via_extra_body(compat_client, text_model_id):
    """Test that shields parameter is received by the server and raises NotImplementedError."""

    # Test with shields as list of strings (shield IDs)
    with pytest.raises((APIStatusError, NotImplementedError)) as exc_info:
        compat_client.responses.create(
            model=text_model_id,
            input="What is the capital of France?",
            stream=False,
            extra_body={"shields": ["test-shield-1", "test-shield-2"]},
        )

    # Verify the error message indicates shields are not implemented
    error_message = str(exc_info.value)
    assert "not yet implemented" in error_message.lower() or "not implemented" in error_message.lower()
