# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Callable, TypeVar

from llama_stack.core.datatypes import (
    InferenceStoreReference,
    PersistenceConfig,
    StoreReference,
)
from llama_stack.core.utils.config_dirs import DISTRIBS_BASE_DIR, RUNTIME_BASE_DIR
from llama_stack.providers.utils.kvstore.config import KVStoreConfig, SqliteKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import (
    SqlStoreConfig,
    SqliteSqlStoreConfig,
)

T = TypeVar("T", KVStoreConfig, SqlStoreConfig)


def resolve_backend(
    persistence: PersistenceConfig | None,
    store_ref: StoreReference | None,
    default_factory: Callable[[], T],
    store_name: str,
) -> T:
    """
    Resolve a store reference to actual backend config.

    Args:
        persistence: Global persistence config
        store_ref: Store reference (e.g., metadata, inference)
        default_factory: Function to create default config if not specified
        store_name: Name for error messages

    Returns:
        Resolved backend config with store-specific overlays applied
    """
    if not persistence or not persistence.stores or not store_ref:
        return default_factory()

    backend_config = persistence.backends.get(store_ref.backend)
    if not backend_config:
        raise ValueError(
            f"Backend '{store_ref.backend}' referenced by store '{store_name}' "
            f"not found in persistence.backends"
        )

    # Clone backend and apply namespace if KVStore
    if isinstance(backend_config, (KVStoreConfig.__args__)):  # type: ignore
        config_dict = backend_config.model_dump()
        if store_ref.namespace:
            config_dict["namespace"] = store_ref.namespace
        return type(backend_config)(**config_dict)  # type: ignore

    return backend_config  # type: ignore


def resolve_inference_store_config(
    persistence: PersistenceConfig | None,
) -> tuple[SqlStoreConfig, int, int]:
    """
    Resolve inference store configuration.

    Returns:
        (sql_config, max_queue_size, num_writers)
    """
    if not persistence or not persistence.stores or not persistence.stores.inference:
        # Default SQLite
        return (
            SqliteSqlStoreConfig(
                db_path=(RUNTIME_BASE_DIR / "inference.db").as_posix(),
            ),
            10000,
            4,
        )

    inference_ref = persistence.stores.inference
    backend_config = persistence.backends.get(inference_ref.backend)
    if not backend_config:
        raise ValueError(
            f"Backend '{inference_ref.backend}' referenced by inference store "
            f"not found in persistence.backends"
        )

    if not isinstance(backend_config, (SqlStoreConfig.__args__)):  # type: ignore
        raise ValueError(
            f"Inference store requires SqlStore backend, got {type(backend_config).__name__}"
        )

    return (
        backend_config,  # type: ignore
        inference_ref.max_write_queue_size,
        inference_ref.num_writers,
    )


def resolve_metadata_store_config(
    persistence: PersistenceConfig | None,
    image_name: str,
) -> KVStoreConfig:
    """
    Resolve metadata store configuration.

    Args:
        persistence: Global persistence config
        image_name: Distribution image name for default path

    Returns:
        Resolved KVStore config
    """
    default_config = SqliteKVStoreConfig(
        db_path=(DISTRIBS_BASE_DIR / image_name / "kvstore.db").as_posix()
    )

    store_ref = None
    if persistence and persistence.stores:
        store_ref = persistence.stores.metadata

    return resolve_backend(
        persistence=persistence,
        store_ref=store_ref,
        default_factory=lambda: default_config,
        store_name="metadata",
    )


def resolve_conversations_store_config(
    persistence: PersistenceConfig | None,
) -> SqlStoreConfig:
    """
    Resolve conversations store configuration.

    Returns:
        Resolved SqlStore config
    """
    default_config = SqliteSqlStoreConfig(
        db_path=(RUNTIME_BASE_DIR / "conversations.db").as_posix()
    )

    store_ref = None
    if persistence and persistence.stores:
        store_ref = persistence.stores.conversations

    return resolve_backend(
        persistence=persistence,
        store_ref=store_ref,
        default_factory=lambda: default_config,
        store_name="conversations",
    )
