# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import aiohttp

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.common.type_system import ParamType
from llama_stack.apis.datasets import Dataset

from .config import NvidiaDatasetIOConfig


class NvidiaDatasetIOAdapter:
    """Nvidia NeMo DatasetIO API."""

    def __init__(self, config: NvidiaDatasetIOConfig):
        self.config = config
        self.headers = {}

    async def _make_request(
        self,
        method: str,
        path: str,
        headers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Helper method to make HTTP requests to the Customizer API."""
        url = f"{self.config.datasets_url}{path}"
        request_headers = self.headers.copy()

        # Set default Content-Type for JSON requests
        if json is not None:
            request_headers["Content-Type"] = "application/json"

        if headers:
            request_headers.update(headers)

        async with aiohttp.ClientSession(headers=request_headers) as session:
            async with session.request(method, url, params=params, json=json, **kwargs) as response:
                if response.status != 200:
                    error_data = await response.json()
                    raise Exception(f"API request failed: {error_data}")
                return await response.json()

    async def register_dataset(
        self,
        dataset_def: Dataset,
    ) -> Dataset:
        """Register a new dataset.

        Args:
            dataset_def [Dataset]: The dataset definition.
                dataset_id [str]: The ID of the dataset.
                source [DataSource]: The source of the dataset.
                metadata [Dict[str, Any]]: The metadata of the dataset.
                    format [str]: The format of the dataset.
                    description [str]: The description of the dataset.
        Returns:
            Dataset
        """
        # add warnings for unsupported params
        request_body = {
            "name": dataset_def.identifier,
            "namespace": self.config.dataset_namespace,
            "files_url": dataset_def.source.uri,
            "project": self.config.project_id,
        }
        if dataset_def.metadata:
            request_body["format"] = dataset_def.metadata.get("format")
            request_body["description"] = dataset_def.metadata.get("description")
        await self._make_request(
            "POST",
            "/v1/datasets",
            json=request_body,
        )
        return dataset_def

    async def update_dataset(
        self,
        dataset_id: str,
        dataset_schema: dict[str, ParamType],
        url: URL,
        provider_dataset_id: str | None = None,
        provider_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("Not implemented")

    async def unregister_dataset(
        self,
        dataset_id: str,
    ) -> None:
        await self._make_request(
            "DELETE",
            f"/v1/datasets/{self.config.dataset_namespace}/{dataset_id}",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    async def iterrows(
        self,
        dataset_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        raise NotImplementedError("Not implemented")

    async def append_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        raise NotImplementedError("Not implemented")
