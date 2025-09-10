# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any

from llama_stack.apis.common.responses import PaginatedResponse
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Dataset
from llama_stack.providers.datatypes import DatasetsProtocolPrivate
from llama_stack.providers.utils.datasetio.url_utils import get_dataframe_from_uri
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.pagination import paginate_records

from .config import LocalFSDatasetIOConfig

DATASETS_PREFIX = "localfs_datasets:"


class PandasDataframeDataset:
    def __init__(self, dataset_def: Dataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_def = dataset_def
        self.df = None

    def __len__(self) -> int:
        assert self.df is not None, "Dataset not loaded. Please call .load() first"
        return len(self.df)

    def __getitem__(self, idx):
        assert self.df is not None, "Dataset not loaded. Please call .load() first"
        if isinstance(idx, slice):
            return self.df.iloc[idx].to_dict(orient="records")
        else:
            return self.df.iloc[idx].to_dict()

    async def load(self) -> None:
        if self.df is not None:
            return

        if self.dataset_def.source.type == "uri":
            self.df = await get_dataframe_from_uri(self.dataset_def.source.uri)
        elif self.dataset_def.source.type == "rows":
            import pandas

            self.df = pandas.DataFrame(self.dataset_def.source.rows)
        else:
            raise ValueError(f"Unsupported dataset source type: {self.dataset_def.source.type}")

        if self.df is None:
            raise ValueError(f"Failed to load dataset from {self.dataset_def.url}")


class LocalFSDatasetIOImpl(DatasetIO, DatasetsProtocolPrivate):
    def __init__(self, config: LocalFSDatasetIOConfig) -> None:
        self.config = config
        # local registry for keeping track of datasets within the provider
        self.dataset_infos = {}
        self.kvstore = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing datasets from kvstore
        start_key = DATASETS_PREFIX
        end_key = f"{DATASETS_PREFIX}\xff"
        stored_datasets = await self.kvstore.values_in_range(start_key, end_key)

        for dataset in stored_datasets:
            dataset = Dataset.model_validate_json(dataset)
            self.dataset_infos[dataset.identifier] = dataset

    async def shutdown(self) -> None: ...

    async def register_dataset(
        self,
        dataset_def: Dataset,
    ) -> None:
        # Store in kvstore
        key = f"{DATASETS_PREFIX}{dataset_def.identifier}"
        await self.kvstore.set(
            key=key,
            value=dataset_def.model_dump_json(),
        )
        self.dataset_infos[dataset_def.identifier] = dataset_def

    async def unregister_dataset(self, dataset_id: str) -> None:
        key = f"{DATASETS_PREFIX}{dataset_id}"
        await self.kvstore.delete(key=key)
        del self.dataset_infos[dataset_id]

    async def iterrows(
        self,
        dataset_id: str,
        start_index: int | None = None,
        limit: int | None = None,
    ) -> PaginatedResponse:
        dataset_def = self.dataset_infos[dataset_id]
        dataset_impl = PandasDataframeDataset(dataset_def)
        await dataset_impl.load()

        records = dataset_impl.df.to_dict("records")
        return paginate_records(records, start_index, limit)

    async def append_rows(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        import pandas

        dataset_def = self.dataset_infos[dataset_id]
        dataset_impl = PandasDataframeDataset(dataset_def)
        await dataset_impl.load()

        new_rows_df = pandas.DataFrame(rows)
        dataset_impl.df = pandas.concat([dataset_impl.df, new_rows_df], ignore_index=True)
