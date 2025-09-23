# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import boto3
import pytest
from moto import mock_aws

from llama_stack.providers.remote.files.s3 import S3FilesImplConfig, get_adapter_impl
from llama_stack.providers.utils.sqlstore.sqlstore import SqliteSqlStoreConfig


class MockUploadFile:
    def __init__(self, content: bytes, filename: str, content_type: str = "text/plain"):
        self.content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self.content


@pytest.fixture
def sample_text_file():
    content = b"Hello, this is a test file for the S3 Files API!"
    return MockUploadFile(content, "sample_text_file-0.txt")


@pytest.fixture
def sample_text_file2():
    content = b"Hello, this is a second test file for the S3 Files API!"
    return MockUploadFile(content, "sample_text_file-1.txt")


@pytest.fixture
def s3_config(tmp_path):
    db_path = tmp_path / "s3_files_metadata.db"

    return S3FilesImplConfig(
        bucket_name=f"test-bucket-{tmp_path.name}",
        region="not-a-region",
        auto_create_bucket=True,
        metadata_store=SqliteSqlStoreConfig(db_path=db_path.as_posix()),
    )


@pytest.fixture
def s3_client():
    # we use `with mock_aws()` because @mock_aws decorator does not support
    # being a generator
    with mock_aws():
        # must yield or the mock will be reset before it is used
        yield boto3.client("s3")


@pytest.fixture
async def s3_provider(s3_config, s3_client):  # s3_client provides the moto mock, don't remove it
    provider = await get_adapter_impl(s3_config, {})
    yield provider
    await provider.shutdown()
