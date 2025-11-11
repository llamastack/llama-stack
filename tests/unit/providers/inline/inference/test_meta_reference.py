# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import Mock

import pytest

from llama_stack.providers.inline.inference.meta_reference.model_parallel import (
    ModelRunner,
)


class TestModelRunner:
    """Test ModelRunner task dispatching for model-parallel inference."""

    def test_chat_completion_task_dispatch(self):
        """Verify ModelRunner correctly dispatches chat_completion tasks."""
        # Create a mock generator
        mock_generator = Mock()
        mock_generator.chat_completion = Mock(return_value=iter([]))

        runner = ModelRunner(mock_generator)

        # Create a chat_completion task
        fake_params = {"model": "test"}
        fake_messages = [{"role": "user", "content": "test"}]
        task = ("chat_completion", [fake_params, fake_messages])

        # Execute task
        runner(task)

        # Verify chat_completion was called with correct arguments
        mock_generator.chat_completion.assert_called_once_with(fake_params, fake_messages)

    def test_invalid_task_type_raises_error(self):
        """Verify ModelRunner rejects invalid task types."""
        mock_generator = Mock()
        runner = ModelRunner(mock_generator)

        with pytest.raises(ValueError, match="Unexpected task type"):
            runner(("invalid_task", []))
