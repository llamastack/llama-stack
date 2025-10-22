# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.common.errors import ResourceNotFoundError
from llama_stack.apis.common.responses import Order
from llama_stack.apis.files import OpenAIFilePurpose
from llama_stack.core.access_control.access_control import default_policy
from llama_stack.core.storage.datatypes import SqliteSqlStoreConfig, SqlStoreReference
from llama_stack.providers.inline.files.localfs import (
    LocalfsFilesImpl,
    LocalfsFilesImplConfig,
)
from llama_stack.providers.utils.sqlstore.sqlstore import register_sqlstore_backends


class MockUploadFile:
    """Mock UploadFile for testing file uploads."""

    def __init__(self, content: bytes, filename: str, content_type: str = "text/plain"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self.content


@pytest.fixture
async def files_provider(tmp_path):
    """Create a files provider with temporary storage for testing."""
    storage_dir = tmp_path / "files"
    db_path = tmp_path / "files_metadata.db"

    backend_name = "sql_localfs_test"
    register_sqlstore_backends({backend_name: SqliteSqlStoreConfig(db_path=db_path.as_posix())})
    config = LocalfsFilesImplConfig(
        storage_dir=storage_dir.as_posix(),
        metadata_store=SqlStoreReference(backend=backend_name, table_name="files_metadata"),
    )

    provider = LocalfsFilesImpl(config, default_policy())
    await provider.initialize()
    yield provider


@pytest.fixture
def sample_text_file():
    """Sample text file for testing."""
    content = b"Hello, this is a test file for the OpenAI Files API!"
    return MockUploadFile(content, "test.txt", "text/plain")


@pytest.fixture
def sample_json_file():
    """Sample JSON file for testing."""
    content = b'{"message": "Hello, World!", "type": "test"}'
    return MockUploadFile(content, "data.json", "application/json")


@pytest.fixture
def large_file():
    """Large file for testing file size handling."""
    content = b"x" * 1024 * 1024  # 1MB file
    return MockUploadFile(content, "large_file.bin", "application/octet-stream")


class TestOpenAIFilesAPI:
    """Test suite for OpenAI Files API endpoints."""

    async def test_upload_file_success(self, files_provider, sample_text_file):
        """Test successful file upload."""
        # Upload file
        result = await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

        # Verify response
        assert result.id.startswith("file-")
        assert result.filename == "test.txt"
        assert result.purpose == OpenAIFilePurpose.ASSISTANTS
        assert result.bytes == len(sample_text_file.content)
        assert result.created_at > 0
        assert result.expires_at > result.created_at

    async def test_upload_different_purposes(self, files_provider, sample_text_file):
        """Test uploading files with different purposes."""
        purposes = list(OpenAIFilePurpose)

        uploaded_files = []
        for purpose in purposes:
            result = await files_provider.openai_upload_file(file=sample_text_file, purpose=purpose)
            uploaded_files.append(result)
            assert result.purpose == purpose

    async def test_upload_different_file_types(self, files_provider, sample_text_file, sample_json_file, large_file):
        """Test uploading different types and sizes of files."""
        files_to_test = [
            (sample_text_file, "test.txt"),
            (sample_json_file, "data.json"),
            (large_file, "large_file.bin"),
        ]

        for file_obj, expected_filename in files_to_test:
            result = await files_provider.openai_upload_file(file=file_obj, purpose=OpenAIFilePurpose.ASSISTANTS)
            assert result.filename == expected_filename
            assert result.bytes == len(file_obj.content)

    async def test_list_files_empty(self, files_provider):
        """Test listing files when no files exist."""
        result = await files_provider.openai_list_files()

        assert result.data == []
        assert result.has_more is False
        assert result.first_id == ""
        assert result.last_id == ""

    async def test_list_files_with_content(self, files_provider, sample_text_file, sample_json_file):
        """Test listing files when files exist."""
        # Upload multiple files
        file1 = await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)
        file2 = await files_provider.openai_upload_file(file=sample_json_file, purpose=OpenAIFilePurpose.ASSISTANTS)

        # List files
        result = await files_provider.openai_list_files()

        assert len(result.data) == 2
        file_ids = [f.id for f in result.data]
        assert file1.id in file_ids
        assert file2.id in file_ids

    async def test_list_files_with_purpose_filter(self, files_provider, sample_text_file):
        """Test listing files with purpose filtering."""
        # Upload file with specific purpose
        uploaded_file = await files_provider.openai_upload_file(
            file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        # List files with matching purpose
        result = await files_provider.openai_list_files(purpose=OpenAIFilePurpose.ASSISTANTS)
        assert len(result.data) == 1
        assert result.data[0].id == uploaded_file.id
        assert result.data[0].purpose == OpenAIFilePurpose.ASSISTANTS

    async def test_list_files_with_limit(self, files_provider, sample_text_file):
        """Test listing files with limit parameter."""
        # Upload multiple files
        for _ in range(5):
            await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)

        # List with limit
        result = await files_provider.openai_list_files(limit=3)
        assert len(result.data) == 3

    async def test_list_files_with_order(self, files_provider, sample_text_file):
        """Test listing files with different order."""
        # Upload multiple files
        files = []
        for _ in range(3):
            file = await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)
            files.append(file)

        # Test descending order (default)
        result_desc = await files_provider.openai_list_files(order=Order.desc)
        assert len(result_desc.data) == 3
        # Most recent should be first
        assert result_desc.data[0].created_at >= result_desc.data[1].created_at >= result_desc.data[2].created_at

        # Test ascending order
        result_asc = await files_provider.openai_list_files(order=Order.asc)
        assert len(result_asc.data) == 3
        # Oldest should be first
        assert result_asc.data[0].created_at <= result_asc.data[1].created_at <= result_asc.data[2].created_at

    async def test_retrieve_file_success(self, files_provider, sample_text_file):
        """Test successful file retrieval."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        # Retrieve file
        retrieved_file = await files_provider.openai_retrieve_file(uploaded_file.id)

        # Verify response
        assert retrieved_file.id == uploaded_file.id
        assert retrieved_file.filename == uploaded_file.filename
        assert retrieved_file.purpose == uploaded_file.purpose
        assert retrieved_file.bytes == uploaded_file.bytes
        assert retrieved_file.created_at == uploaded_file.created_at
        assert retrieved_file.expires_at == uploaded_file.expires_at

    async def test_retrieve_file_not_found(self, files_provider):
        """Test retrieving a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file("file-nonexistent")

    async def test_retrieve_file_content_success(self, files_provider, sample_text_file):
        """Test successful file content retrieval."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        # Retrieve file content
        content = await files_provider.openai_retrieve_file_content(uploaded_file.id)

        # Verify content
        assert content.body == sample_text_file.content

    async def test_retrieve_file_content_not_found(self, files_provider):
        """Test retrieving content of a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file_content("file-nonexistent")

    async def test_delete_file_success(self, files_provider, sample_text_file):
        """Test successful file deletion."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        # Verify file exists
        await files_provider.openai_retrieve_file(uploaded_file.id)

        # Delete file
        delete_response = await files_provider.openai_delete_file(uploaded_file.id)

        # Verify delete response
        assert delete_response.id == uploaded_file.id
        assert delete_response.deleted is True

        # Verify file no longer exists
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_retrieve_file(uploaded_file.id)

    async def test_delete_file_not_found(self, files_provider):
        """Test deleting a non-existent file."""
        with pytest.raises(ResourceNotFoundError, match="not found"):
            await files_provider.openai_delete_file("file-nonexistent")

    async def test_file_persistence_across_operations(self, files_provider, sample_text_file):
        """Test that files persist correctly across multiple operations."""
        # Upload file
        uploaded_file = await files_provider.openai_upload_file(
            file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        # Verify it appears in listing
        files_list = await files_provider.openai_list_files()
        assert len(files_list.data) == 1
        assert files_list.data[0].id == uploaded_file.id

        # Retrieve file info
        retrieved_file = await files_provider.openai_retrieve_file(uploaded_file.id)
        assert retrieved_file.id == uploaded_file.id

        # Retrieve file content
        content = await files_provider.openai_retrieve_file_content(uploaded_file.id)
        assert content.body == sample_text_file.content

        # Delete file
        await files_provider.openai_delete_file(uploaded_file.id)

        # Verify it's gone from listing
        files_list = await files_provider.openai_list_files()
        assert len(files_list.data) == 0

    async def test_multiple_files_operations(self, files_provider, sample_text_file, sample_json_file):
        """Test operations with multiple files."""
        # Upload multiple files
        file1 = await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)
        file2 = await files_provider.openai_upload_file(file=sample_json_file, purpose=OpenAIFilePurpose.ASSISTANTS)

        # Verify both exist
        files_list = await files_provider.openai_list_files()
        assert len(files_list.data) == 2

        # Delete one file
        await files_provider.openai_delete_file(file1.id)

        # Verify only one remains
        files_list = await files_provider.openai_list_files()
        assert len(files_list.data) == 1
        assert files_list.data[0].id == file2.id

        # Verify the remaining file is still accessible
        content = await files_provider.openai_retrieve_file_content(file2.id)
        assert content.body == sample_json_file.content

    async def test_file_id_uniqueness(self, files_provider, sample_text_file):
        """Test that each uploaded file gets a unique ID."""
        file_ids = set()

        # Upload same file multiple times
        for _ in range(10):
            uploaded_file = await files_provider.openai_upload_file(
                file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS
            )
            assert uploaded_file.id not in file_ids, f"Duplicate file ID: {uploaded_file.id}"
            file_ids.add(uploaded_file.id)
            assert uploaded_file.id.startswith("file-")

    async def test_file_no_filename_handling(self, files_provider):
        """Test handling files with no filename."""
        file_without_name = MockUploadFile(b"content", None)  # No filename

        uploaded_file = await files_provider.openai_upload_file(
            file=file_without_name, purpose=OpenAIFilePurpose.ASSISTANTS
        )

        assert uploaded_file.filename == "uploaded_file"  # Default filename

    async def test_after_pagination_works(self, files_provider, sample_text_file):
        """Test that 'after' pagination works correctly."""
        # Upload multiple files to test pagination
        uploaded_files = []
        for _ in range(5):
            file = await files_provider.openai_upload_file(file=sample_text_file, purpose=OpenAIFilePurpose.ASSISTANTS)
            uploaded_files.append(file)

        # Get first page without 'after' parameter
        first_page = await files_provider.openai_list_files(limit=2, order=Order.desc)
        assert len(first_page.data) == 2
        assert first_page.has_more is True

        # Get second page using 'after' parameter
        second_page = await files_provider.openai_list_files(after=first_page.data[-1].id, limit=2, order=Order.desc)
        assert len(second_page.data) <= 2

        # Verify no overlap between pages
        first_page_ids = {f.id for f in first_page.data}
        second_page_ids = {f.id for f in second_page.data}
        assert first_page_ids.isdisjoint(second_page_ids)
