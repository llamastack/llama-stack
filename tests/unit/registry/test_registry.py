# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.apis.inference import Model
from llama_stack.apis.vector_stores import VectorStore
from llama_stack.core.datatypes import VectorStoreWithOwner
from llama_stack.core.storage.datatypes import KVStoreReference, SqliteKVStoreConfig
from llama_stack.core.store.registry import (
    KEY_FORMAT,
    CachedDiskDistributionRegistry,
    DiskDistributionRegistry,
)
from llama_stack.providers.utils.kvstore import kvstore_impl, register_kvstore_backends


@pytest.fixture
def sample_vector_store():
    return VectorStore(
        identifier="test_vector_store",
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimension=768,
        provider_resource_id="test_vector_store",
        provider_id="test-provider",
    )


@pytest.fixture
def sample_model():
    return Model(
        identifier="test_model",
        provider_resource_id="test_model",
        provider_id="test-provider",
    )


async def test_registry_initialization(disk_dist_registry):
    # Test empty registry
    result = await disk_dist_registry.get("nonexistent", "nonexistent")
    assert result is None


async def test_basic_registration(disk_dist_registry, sample_vector_store, sample_model):
    print(f"Registering {sample_vector_store}")
    await disk_dist_registry.register(sample_vector_store)
    print(f"Registering {sample_model}")
    await disk_dist_registry.register(sample_model)
    print("Getting vector_store")
    result_vector_store = await disk_dist_registry.get("vector_store", "test_vector_store")
    assert result_vector_store is not None
    assert result_vector_store.identifier == sample_vector_store.identifier
    assert result_vector_store.embedding_model == sample_vector_store.embedding_model
    assert result_vector_store.provider_id == sample_vector_store.provider_id

    result_model = await disk_dist_registry.get("model", "test_model")
    assert result_model is not None
    assert result_model.identifier == sample_model.identifier
    assert result_model.provider_id == sample_model.provider_id


async def test_cached_registry_initialization(sqlite_kvstore, sample_vector_store, sample_model):
    # First populate the disk registry
    disk_registry = DiskDistributionRegistry(sqlite_kvstore)
    await disk_registry.initialize()
    await disk_registry.register(sample_vector_store)
    await disk_registry.register(sample_model)

    # Test cached version loads from disk
    db_path = sqlite_kvstore.db_path
    backend_name = "kv_cached_test"
    register_kvstore_backends({backend_name: SqliteKVStoreConfig(db_path=db_path)})
    cached_registry = CachedDiskDistributionRegistry(
        await kvstore_impl(KVStoreReference(backend=backend_name, namespace="registry"))
    )
    await cached_registry.initialize()

    result_vector_store = await cached_registry.get("vector_store", "test_vector_store")
    assert result_vector_store is not None
    assert result_vector_store.identifier == sample_vector_store.identifier
    assert result_vector_store.embedding_model == sample_vector_store.embedding_model
    assert result_vector_store.embedding_dimension == sample_vector_store.embedding_dimension
    assert result_vector_store.provider_id == sample_vector_store.provider_id


async def test_cached_registry_updates(cached_disk_dist_registry):
    new_vector_store = VectorStore(
        identifier="test_vector_store_2",
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimension=768,
        provider_resource_id="test_vector_store_2",
        provider_id="baz",
    )
    await cached_disk_dist_registry.register(new_vector_store)

    # Verify in cache
    result_vector_store = await cached_disk_dist_registry.get("vector_store", "test_vector_store_2")
    assert result_vector_store is not None
    assert result_vector_store.identifier == new_vector_store.identifier
    assert result_vector_store.provider_id == new_vector_store.provider_id

    # Verify persisted to disk
    db_path = cached_disk_dist_registry.kvstore.db_path
    backend_name = "kv_cached_new"
    register_kvstore_backends({backend_name: SqliteKVStoreConfig(db_path=db_path)})
    new_registry = DiskDistributionRegistry(
        await kvstore_impl(KVStoreReference(backend=backend_name, namespace="registry"))
    )
    await new_registry.initialize()
    result_vector_store = await new_registry.get("vector_store", "test_vector_store_2")
    assert result_vector_store is not None
    assert result_vector_store.identifier == new_vector_store.identifier
    assert result_vector_store.provider_id == new_vector_store.provider_id


async def test_duplicate_provider_registration(cached_disk_dist_registry):
    original_vector_store = VectorStore(
        identifier="test_vector_store_2",
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimension=768,
        provider_resource_id="test_vector_store_2",
        provider_id="baz",
    )
    assert await cached_disk_dist_registry.register(original_vector_store)

    duplicate_vector_store = VectorStore(
        identifier="test_vector_store_2",
        embedding_model="different-model",
        embedding_dimension=768,
        provider_resource_id="test_vector_store_2",
        provider_id="baz",  # Same provider_id
    )
    with pytest.raises(
        ValueError, match="Object of type 'vector_store' and identifier 'test_vector_store_2' already exists"
    ):
        await cached_disk_dist_registry.register(duplicate_vector_store)

    result = await cached_disk_dist_registry.get("vector_store", "test_vector_store_2")
    assert result is not None
    assert result.embedding_model == original_vector_store.embedding_model  # Original values preserved


async def test_get_all_objects(cached_disk_dist_registry):
    # Create multiple test banks
    # Create multiple test banks
    test_vector_stores = [
        VectorStore(
            identifier=f"test_vector_store_{i}",
            embedding_model="nomic-embed-text-v1.5",
            embedding_dimension=768,
            provider_resource_id=f"test_vector_store_{i}",
            provider_id=f"provider_{i}",
        )
        for i in range(3)
    ]

    # Register all vector_stores
    for vector_store in test_vector_stores:
        await cached_disk_dist_registry.register(vector_store)

    # Test get_all retrieval
    all_results = await cached_disk_dist_registry.get_all()
    assert len(all_results) == 3

    # Verify each vector_store was stored correctly
    for original_vector_store in test_vector_stores:
        matching_vector_stores = [v for v in all_results if v.identifier == original_vector_store.identifier]
        assert len(matching_vector_stores) == 1
        stored_vector_store = matching_vector_stores[0]
        assert stored_vector_store.embedding_model == original_vector_store.embedding_model
        assert stored_vector_store.provider_id == original_vector_store.provider_id
        assert stored_vector_store.embedding_dimension == original_vector_store.embedding_dimension


async def test_parse_registry_values_error_handling(sqlite_kvstore):
    valid_db = VectorStore(
        identifier="valid_vector_store",
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimension=768,
        provider_resource_id="valid_vector_store",
        provider_id="test-provider",
    )

    await sqlite_kvstore.set(
        KEY_FORMAT.format(type="vector_store", identifier="valid_vector_store"), valid_db.model_dump_json()
    )

    await sqlite_kvstore.set(KEY_FORMAT.format(type="vector_store", identifier="corrupted_json"), "{not valid json")

    await sqlite_kvstore.set(
        KEY_FORMAT.format(type="vector_store", identifier="missing_fields"),
        '{"type": "vector_store", "identifier": "missing_fields"}',
    )

    test_registry = DiskDistributionRegistry(sqlite_kvstore)
    await test_registry.initialize()

    # Get all objects, which should only return the valid one
    all_objects = await test_registry.get_all()

    # Should have filtered out the invalid entries
    assert len(all_objects) == 1
    assert all_objects[0].identifier == "valid_vector_store"

    # Check that the get method also handles errors correctly
    invalid_obj = await test_registry.get("vector_store", "corrupted_json")
    assert invalid_obj is None

    invalid_obj = await test_registry.get("vector_store", "missing_fields")
    assert invalid_obj is None


async def test_cached_registry_error_handling(sqlite_kvstore):
    valid_db = VectorStore(
        identifier="valid_cached_db",
        embedding_model="nomic-embed-text-v1.5",
        embedding_dimension=768,
        provider_resource_id="valid_cached_db",
        provider_id="test-provider",
    )

    await sqlite_kvstore.set(
        KEY_FORMAT.format(type="vector_store", identifier="valid_cached_db"), valid_db.model_dump_json()
    )

    await sqlite_kvstore.set(
        KEY_FORMAT.format(type="vector_store", identifier="invalid_cached_db"),
        '{"type": "vector_store", "identifier": "invalid_cached_db", "embedding_model": 12345}',  # Should be string
    )

    cached_registry = CachedDiskDistributionRegistry(sqlite_kvstore)
    await cached_registry.initialize()

    all_objects = await cached_registry.get_all()
    assert len(all_objects) == 1
    assert all_objects[0].identifier == "valid_cached_db"

    invalid_obj = await cached_registry.get("vector_store", "invalid_cached_db")
    assert invalid_obj is None


async def test_double_registration_identical_objects(disk_dist_registry):
    """Test that registering identical objects succeeds (idempotent)."""
    vector_store = VectorStoreWithOwner(
        identifier="test_vector_store",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="test_vector_store",
        provider_id="test-provider",
    )

    # First registration should succeed
    result1 = await disk_dist_registry.register(vector_store)
    assert result1 is True

    # Second registration of identical object should also succeed (idempotent)
    result2 = await disk_dist_registry.register(vector_store)
    assert result2 is True

    # Verify object exists and is unchanged
    retrieved = await disk_dist_registry.get("vector_store", "test_vector_store")
    assert retrieved is not None
    assert retrieved.identifier == vector_store.identifier
    assert retrieved.embedding_model == vector_store.embedding_model


async def test_double_registration_different_objects(disk_dist_registry):
    """Test that registering different objects with same identifier fails."""
    vector_store1 = VectorStoreWithOwner(
        identifier="test_vector_store",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="test_vector_store",
        provider_id="test-provider",
    )

    vector_store2 = VectorStoreWithOwner(
        identifier="test_vector_store",  # Same identifier
        embedding_model="different-model",  # Different embedding model
        embedding_dimension=384,
        provider_resource_id="test_vector_store",
        provider_id="test-provider",
    )

    # First registration should succeed
    result1 = await disk_dist_registry.register(vector_store1)
    assert result1 is True

    # Second registration with different data should fail
    with pytest.raises(
        ValueError, match="Object of type 'vector_store' and identifier 'test_vector_store' already exists"
    ):
        await disk_dist_registry.register(vector_store2)

    # Verify original object is unchanged
    retrieved = await disk_dist_registry.get("vector_store", "test_vector_store")
    assert retrieved is not None
    assert retrieved.embedding_model == "all-MiniLM-L6-v2"  # Original value


async def test_double_registration_with_cache(cached_disk_dist_registry):
    """Test double registration behavior with caching enabled."""
    from llama_stack.apis.models import ModelType
    from llama_stack.core.datatypes import ModelWithOwner

    model1 = ModelWithOwner(
        identifier="test_model",
        provider_resource_id="test_model",
        provider_id="test-provider",
        model_type=ModelType.llm,
    )

    model2 = ModelWithOwner(
        identifier="test_model",  # Same identifier
        provider_resource_id="test_model",
        provider_id="test-provider",
        model_type=ModelType.embedding,  # Different type
    )

    # First registration should succeed and populate cache
    result1 = await cached_disk_dist_registry.register(model1)
    assert result1 is True

    # Verify in cache
    cached_model = cached_disk_dist_registry.get_cached("model", "test_model")
    assert cached_model is not None
    assert cached_model.model_type == ModelType.llm

    # Second registration with different data should fail
    with pytest.raises(ValueError, match="Object of type 'model' and identifier 'test_model' already exists"):
        await cached_disk_dist_registry.register(model2)

    # Cache should still contain original model
    cached_model_after = cached_disk_dist_registry.get_cached("model", "test_model")
    assert cached_model_after is not None
    assert cached_model_after.model_type == ModelType.llm
