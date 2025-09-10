# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

import pytest

from llama_stack.apis.datasets import Dataset, DatasetPurpose, URIDataSource
from llama_stack.apis.resource import ResourceType
from llama_stack.providers.remote.datasetio.nvidia.config import NvidiaDatasetIOConfig
from llama_stack.providers.remote.datasetio.nvidia.datasetio import NvidiaDatasetIOAdapter


@pytest.fixture
def nvidia_adapter():
    """Fixture to set up NvidiaDatasetIOAdapter with mocked requests."""
    os.environ["NVIDIA_DATASETS_URL"] = "http://nemo.test/datasets"

    config = NvidiaDatasetIOConfig(
        datasets_url=os.environ["NVIDIA_DATASETS_URL"], dataset_namespace="default", project_id="default"
    )
    adapter = NvidiaDatasetIOAdapter(config)

    with patch(
        "llama_stack.providers.remote.datasetio.nvidia.datasetio.NvidiaDatasetIOAdapter._make_request"
    ) as mock_make_request:
        yield adapter, mock_make_request


def _assert_request(mock_call, expected_method, expected_path, expected_json=None):
    """Helper function to verify request details in mock calls."""
    call_args = mock_call.call_args

    assert call_args[0][0] == expected_method
    assert call_args[0][1] == expected_path

    if expected_json:
        for key, value in expected_json.items():
            assert call_args[1]["json"][key] == value


def test_register_dataset(nvidia_adapter, run_async):
    adapter, mock_make_request = nvidia_adapter
    mock_make_request.return_value = {
        "id": "dataset-123456",
        "name": "test-dataset",
        "namespace": "default",
    }

    dataset_def = Dataset(
        identifier="test-dataset",
        type=ResourceType.dataset,
        provider_resource_id="",
        provider_id="",
        purpose=DatasetPurpose.post_training_messages,
        source=URIDataSource(uri="https://example.com/data.jsonl"),
        metadata={"provider_id": "nvidia", "format": "jsonl", "description": "Test dataset description"},
    )

    run_async(adapter.register_dataset(dataset_def))

    mock_make_request.assert_called_once()
    _assert_request(
        mock_make_request,
        "POST",
        "/v1/datasets",
        expected_json={
            "name": "test-dataset",
            "namespace": "default",
            "files_url": "https://example.com/data.jsonl",
            "project": "default",
            "format": "jsonl",
            "description": "Test dataset description",
        },
    )


def test_unregister_dataset(nvidia_adapter, run_async):
    adapter, mock_make_request = nvidia_adapter
    mock_make_request.return_value = {
        "message": "Resource deleted successfully.",
        "id": "dataset-81RSQp7FKX3rdBtKvF9Skn",
        "deleted_at": None,
    }
    dataset_id = "test-dataset"

    run_async(adapter.unregister_dataset(dataset_id))

    mock_make_request.assert_called_once()
    _assert_request(mock_make_request, "DELETE", "/v1/datasets/default/test-dataset")


def test_register_dataset_with_custom_namespace_project(run_async):
    """Test with custom namespace and project configuration."""
    os.environ["NVIDIA_DATASETS_URL"] = "http://nemo.test/datasets"

    custom_config = NvidiaDatasetIOConfig(
        datasets_url=os.environ["NVIDIA_DATASETS_URL"],
        dataset_namespace="custom-namespace",
        project_id="custom-project",
    )
    custom_adapter = NvidiaDatasetIOAdapter(custom_config)

    with patch(
        "llama_stack.providers.remote.datasetio.nvidia.datasetio.NvidiaDatasetIOAdapter._make_request"
    ) as mock_make_request:
        mock_make_request.return_value = {
            "id": "dataset-123456",
            "name": "test-dataset",
            "namespace": "custom-namespace",
        }

        dataset_def = Dataset(
            identifier="test-dataset",
            type=ResourceType.dataset,
            provider_resource_id="",
            provider_id="",
            purpose=DatasetPurpose.post_training_messages,
            source=URIDataSource(uri="https://example.com/data.jsonl"),
            metadata={"format": "jsonl"},
        )

        run_async(custom_adapter.register_dataset(dataset_def))

        mock_make_request.assert_called_once()
        _assert_request(
            mock_make_request,
            "POST",
            "/v1/datasets",
            expected_json={
                "name": "test-dataset",
                "namespace": "custom-namespace",
                "files_url": "https://example.com/data.jsonl",
                "project": "custom-project",
                "format": "jsonl",
            },
        )
