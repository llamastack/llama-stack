# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import patch

import pytest

from llama_stack.apis.common.errors import ResourceNotFoundError
from llama_stack.apis.files import OpenAIFilePurpose
from llama_stack.core.datatypes import User
from llama_stack.providers.remote.files.s3.files import S3FilesImpl


async def test_listing_hides_other_users_file(s3_provider, sample_text_file):
    """Listing should not show files uploaded by other users."""
    user_a = User("user-a", {"roles": ["team-a"]})
    user_b = User("user-b", {"roles": ["team-b"]})

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        listed = await s3_provider.openai_list_files()
        assert all(f.id != uploaded.id for f in listed.data)


@pytest.mark.parametrize(
    "op",
    [S3FilesImpl.openai_retrieve_file, S3FilesImpl.openai_retrieve_file_content, S3FilesImpl.openai_delete_file],
    ids=["retrieve", "content", "delete"],
)
async def test_cannot_access_other_user_file(s3_provider, sample_text_file, op):
    """Operations (metadata/content/delete) on another user's file should raise ResourceNotFoundError.

    `op` is an async callable (provider, file_id) -> awaits the requested operation.
    """
    user_a = User("user-a", {"roles": ["team-a"]})
    user_b = User("user-b", {"roles": ["team-b"]})

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        with pytest.raises(ResourceNotFoundError):
            await op(s3_provider, uploaded.id)


async def test_shared_role_allows_listing(s3_provider, sample_text_file):
    """Listing should show files uploaded by other users when roles are shared."""
    user_a = User("user-a", {"roles": ["shared-role"]})
    user_b = User("user-b", {"roles": ["shared-role"]})

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_a
        uploaded = await s3_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_b
        listed = await s3_provider.openai_list_files()
        assert any(f.id == uploaded.id for f in listed.data)


@pytest.mark.parametrize(
    "op",
    [S3FilesImpl.openai_retrieve_file, S3FilesImpl.openai_retrieve_file_content, S3FilesImpl.openai_delete_file],
    ids=["retrieve", "content", "delete"],
)
async def test_shared_role_allows_access(s3_provider, sample_text_file, op):
    """Operations (metadata/content/delete) on another user's file should succeed when users share a role.

    `op` is an async callable (provider, file_id) -> awaits the requested operation.
    """
    user_x = User("user-x", {"roles": ["shared-role"]})
    user_y = User("user-y", {"roles": ["shared-role"]})

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_x
        uploaded = await s3_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

    with patch("llama_stack.providers.utils.sqlstore.authorized_sqlstore.get_authenticated_user") as mock_get_user:
        mock_get_user.return_value = user_y
        await op(s3_provider, uploaded.id)
