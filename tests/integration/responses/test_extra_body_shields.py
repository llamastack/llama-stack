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
            extra_body={
                "shields": ["test-shield-1", "test-shield-2"]
            }
        )

    # Verify the error message indicates shields are not implemented
    error_message = str(exc_info.value)
    assert "not yet implemented" in error_message.lower() or "not implemented" in error_message.lower()




def test_response_without_shields_still_works(compat_client, text_model_id):
    """Test that responses still work without shields parameter (backwards compatibility)."""

    response = compat_client.responses.create(
        model=text_model_id,
        input="Hello, world!",
        stream=False,
    )

    # Verify response was created successfully
    assert response.id is not None
    assert response.output_text is not None
    assert len(response.output_text) > 0


def test_shields_parameter_received_end_to_end(compat_client, text_model_id):
    """
    Test that shields parameter passed via extra_body reaches the server implementation.

    This verifies end-to-end that:
    1. The parameter can be passed via extra_body in the client SDK
    2. The parameter is properly routed through the API layers
    3. The server-side implementation receives the parameter (verified by NotImplementedError)

    The NotImplementedError proves the extra_body parameter reached the implementation,
    as opposed to being rejected earlier due to signature mismatch or validation errors.
    """
    # Test with shields parameter via extra_body
    with pytest.raises((APIStatusError, NotImplementedError)) as exc_info:
        compat_client.responses.create(
            model=text_model_id,
            input="Test message for shields verification",
            stream=False,
            extra_body={
                "shields": ["shield-1", "shield-2"]
            }
        )

    # The NotImplementedError proves that:
    # 1. extra_body.shields was parsed and passed to the API
    # 2. The server-side implementation received the shields parameter
    # 3. No signature mismatch or validation errors occurred
    error_message = str(exc_info.value)
    assert "not yet implemented" in error_message.lower() or "not implemented" in error_message.lower()
