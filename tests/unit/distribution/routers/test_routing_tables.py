# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Unit tests for the routing tables

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.errors import ModelNotFoundError
from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.datasets.datasets import Dataset, DatasetPurpose, URIDataSource
from llama_stack.apis.datatypes import Api
from llama_stack.apis.models import Model, ModelType
from llama_stack.apis.shields.shields import Shield
from llama_stack.apis.tools import ListToolDefsResponse, ToolDef, ToolGroup
from llama_stack.core.datatypes import RegistryEntrySource
from llama_stack.core.routing_tables.benchmarks import BenchmarksRoutingTable
from llama_stack.core.routing_tables.datasets import DatasetsRoutingTable
from llama_stack.core.routing_tables.models import ModelsRoutingTable
from llama_stack.core.routing_tables.scoring_functions import ScoringFunctionsRoutingTable
from llama_stack.core.routing_tables.shields import ShieldsRoutingTable
from llama_stack.core.routing_tables.toolgroups import ToolGroupsRoutingTable


class Impl:
    def __init__(self, api: Api):
        self.api = api

    @property
    def __provider_spec__(self):
        _provider_spec = AsyncMock()
        _provider_spec.api = self.api
        return _provider_spec


class InferenceImpl(Impl):
    def __init__(self):
        super().__init__(Api.inference)

    async def register_model(self, model: Model):
        return model

    async def unregister_model(self, model_id: str):
        return model_id

    async def should_refresh_models(self):
        return False

    async def list_models(self):
        return [
            Model(
                identifier="provider-model-1",
                provider_resource_id="provider-model-1",
                provider_id="test_provider",
                metadata={},
                model_type=ModelType.llm,
            ),
            Model(
                identifier="provider-model-2",
                provider_resource_id="provider-model-2",
                provider_id="test_provider",
                metadata={"embedding_dimension": 512},
                model_type=ModelType.embedding,
            ),
        ]

    async def shutdown(self):
        pass


class SafetyImpl(Impl):
    def __init__(self):
        super().__init__(Api.safety)

    async def register_shield(self, shield: Shield):
        return shield

    async def unregister_shield(self, shield_id: str):
        return shield_id


class DatasetsImpl(Impl):
    def __init__(self):
        super().__init__(Api.datasetio)

    async def register_dataset(self, dataset: Dataset):
        return dataset

    async def unregister_dataset(self, dataset_id: str):
        return dataset_id


class ScoringFunctionsImpl(Impl):
    def __init__(self):
        super().__init__(Api.scoring)

    async def list_scoring_functions(self):
        return []

    async def register_scoring_function(self, scoring_fn):
        return scoring_fn

    async def unregister_scoring_function(self, scoring_fn_id: str):
        return scoring_fn_id


class BenchmarksImpl(Impl):
    def __init__(self):
        super().__init__(Api.eval)

    async def register_benchmark(self, benchmark):
        return benchmark

    async def unregister_benchmark(self, benchmark_id: str):
        return benchmark_id


class ToolGroupsImpl(Impl):
    def __init__(self):
        super().__init__(Api.tool_runtime)

    async def register_toolgroup(self, toolgroup: ToolGroup):
        return toolgroup

    async def unregister_toolgroup(self, toolgroup_id: str):
        return toolgroup_id

    async def list_runtime_tools(self, toolgroup_id, mcp_endpoint):
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="test-tool",
                    description="Test tool",
                    input_schema={
                        "type": "object",
                        "properties": {"test-param": {"type": "string", "description": "Test param"}},
                    },
                )
            ]
        )


async def test_models_routing_table(cached_disk_dist_registry):
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple models and verify listing
    await table.register_model(model_id="test-model", provider_id="test_provider")
    await table.register_model(model_id="test-model-2", provider_id="test_provider")

    models = await table.list_models()
    assert len(models.data) == 2
    model_ids = {m.identifier for m in models.data}
    assert "test_provider/test-model" in model_ids
    assert "test_provider/test-model-2" in model_ids

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 2
    openai_model_ids = {m.id for m in openai_models.data}
    assert "test_provider/test-model" in openai_model_ids
    assert "test_provider/test-model-2" in openai_model_ids

    # Verify custom_metadata is populated with Llama Stack-specific data
    for openai_model in openai_models.data:
        assert openai_model.custom_metadata is not None
        assert "model_type" in openai_model.custom_metadata
        assert "provider_id" in openai_model.custom_metadata
        assert "provider_resource_id" in openai_model.custom_metadata
        assert openai_model.custom_metadata["provider_id"] == "test_provider"

    # Test get_object_by_identifier
    model = await table.get_object_by_identifier("model", "test_provider/test-model")
    assert model is not None
    assert model.identifier == "test_provider/test-model"

    # Test get_object_by_identifier on non-existent object
    non_existent = await table.get_object_by_identifier("model", "non-existent-model")
    assert non_existent is None

    # Test has_model
    assert await table.has_model("test_provider/test-model")
    assert await table.has_model("test_provider/test-model-2")
    assert not await table.has_model("non-existent-model")
    assert not await table.has_model("test_provider/non-existent-model")

    await table.unregister_model(model_id="test_provider/test-model")
    await table.unregister_model(model_id="test_provider/test-model-2")

    models = await table.list_models()
    assert len(models.data) == 0

    # Test openai list models
    openai_models = await table.openai_list_models()
    assert len(openai_models.data) == 0


async def test_shields_routing_table(cached_disk_dist_registry):
    table = ShieldsRoutingTable({"test_provider": SafetyImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple shields and verify listing
    await table.register_shield(shield_id="test-shield", provider_id="test_provider")
    await table.register_shield(shield_id="test-shield-2", provider_id="test_provider")
    shields = await table.list_shields()
    assert len(shields.data) == 2

    shield_ids = {s.identifier for s in shields.data}
    assert "test-shield" in shield_ids
    assert "test-shield-2" in shield_ids

    # Test get specific shield
    test_shield = await table.get_shield(identifier="test-shield")
    assert test_shield is not None
    assert test_shield.identifier == "test-shield"
    assert test_shield.provider_id == "test_provider"
    assert test_shield.provider_resource_id == "test-shield"
    assert test_shield.params == {}

    # Test get non-existent shield - should raise ValueError with specific message
    with pytest.raises(ValueError, match="Shield 'non-existent' not found"):
        await table.get_shield(identifier="non-existent")

    # Test unregistering shields
    await table.unregister_shield(identifier="test-shield")
    shields = await table.list_shields()

    assert len(shields.data) == 1
    shield_ids = {s.identifier for s in shields.data}
    assert "test-shield" not in shield_ids
    assert "test-shield-2" in shield_ids

    # Unregister the remaining shield
    await table.unregister_shield(identifier="test-shield-2")
    shields = await table.list_shields()
    assert len(shields.data) == 0

    # Test unregistering non-existent shield - should raise ValueError with specific message
    with pytest.raises(ValueError, match="Shield 'non-existent' not found"):
        await table.unregister_shield(identifier="non-existent")


async def test_datasets_routing_table(cached_disk_dist_registry):
    table = DatasetsRoutingTable({"localfs": DatasetsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple datasets and verify listing
    await table.register_dataset(
        dataset_id="test-dataset", purpose=DatasetPurpose.eval_messages_answer, source=URIDataSource(uri="test-uri")
    )
    await table.register_dataset(
        dataset_id="test-dataset-2", purpose=DatasetPurpose.eval_messages_answer, source=URIDataSource(uri="test-uri-2")
    )
    datasets = await table.list_datasets()

    assert len(datasets.data) == 2
    dataset_ids = {d.identifier for d in datasets.data}
    assert "test-dataset" in dataset_ids
    assert "test-dataset-2" in dataset_ids

    await table.unregister_dataset(dataset_id="test-dataset")
    await table.unregister_dataset(dataset_id="test-dataset-2")

    datasets = await table.list_datasets()
    assert len(datasets.data) == 0


async def test_scoring_functions_routing_table(cached_disk_dist_registry):
    table = ScoringFunctionsRoutingTable({"test_provider": ScoringFunctionsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple scoring functions and verify listing
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn",
        provider_id="test_provider",
        description="Test scoring function",
        return_type=NumberType(),
    )
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn-2",
        provider_id="test_provider",
        description="Another test scoring function",
        return_type=NumberType(),
    )
    scoring_functions = await table.list_scoring_functions()

    assert len(scoring_functions.data) == 2
    scoring_fn_ids = {fn.identifier for fn in scoring_functions.data}
    assert "test-scoring-fn" in scoring_fn_ids
    assert "test-scoring-fn-2" in scoring_fn_ids

    # Unregister scoring functions and verify listing
    for i in range(len(scoring_functions.data)):
        await table.unregister_scoring_function(scoring_functions.data[i].scoring_fn_id)

    scoring_functions_list_after_deletion = await table.list_scoring_functions()
    assert len(scoring_functions_list_after_deletion.data) == 0


async def test_double_registration_models_positive(cached_disk_dist_registry):
    """Test that registering the same model twice with identical data succeeds."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a model
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Register the exact same model again - should succeed (idempotent)
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Verify only one model exists
    models = await table.list_models()
    assert len(models.data) == 1
    assert models.data[0].identifier == "test_provider/test-model"


async def test_double_registration_models_negative(cached_disk_dist_registry):
    """Test that registering the same model with different data fails."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a model with specific metadata
    await table.register_model(model_id="test-model", provider_id="test_provider", metadata={"param1": "value1"})

    # Try to register the same model with different metadata - should fail
    with pytest.raises(
        ValueError, match="Object of type 'model' and identifier 'test_provider/test-model' already exists"
    ):
        await table.register_model(
            model_id="test-model", provider_id="test_provider", metadata={"param1": "different_value"}
        )


async def test_double_registration_scoring_functions_positive(cached_disk_dist_registry):
    """Test that registering the same scoring function twice with identical data succeeds."""
    table = ScoringFunctionsRoutingTable({"test_provider": ScoringFunctionsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a scoring function
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn",
        provider_id="test_provider",
        description="Test scoring function",
        return_type=NumberType(),
    )

    # Register the exact same scoring function again - should succeed (idempotent)
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn",
        provider_id="test_provider",
        description="Test scoring function",
        return_type=NumberType(),
    )

    # Verify only one scoring function exists
    scoring_functions = await table.list_scoring_functions()
    assert len(scoring_functions.data) == 1
    assert scoring_functions.data[0].identifier == "test-scoring-fn"


async def test_double_registration_scoring_functions_negative(cached_disk_dist_registry):
    """Test that registering the same scoring function with different data fails."""
    table = ScoringFunctionsRoutingTable({"test_provider": ScoringFunctionsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a scoring function
    await table.register_scoring_function(
        scoring_fn_id="test-scoring-fn",
        provider_id="test_provider",
        description="Test scoring function",
        return_type=NumberType(),
    )

    # Try to register the same scoring function with different description - should fail
    with pytest.raises(
        ValueError, match="Object of type 'scoring_function' and identifier 'test-scoring-fn' already exists"
    ):
        await table.register_scoring_function(
            scoring_fn_id="test-scoring-fn",
            provider_id="test_provider",
            description="Different description",
            return_type=NumberType(),
        )


async def test_double_registration_different_providers(cached_disk_dist_registry):
    """Test that registering objects with same ID but different providers succeeds."""
    impl1 = InferenceImpl()
    impl2 = InferenceImpl()
    table = ModelsRoutingTable({"provider1": impl1, "provider2": impl2}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register same model ID with different providers - should succeed
    await table.register_model(model_id="shared-model", provider_id="provider1")
    await table.register_model(model_id="shared-model", provider_id="provider2")

    # Verify both models exist with different identifiers
    models = await table.list_models()
    assert len(models.data) == 2
    model_ids = {m.identifier for m in models.data}
    assert "provider1/shared-model" in model_ids
    assert "provider2/shared-model" in model_ids


async def test_benchmarks_routing_table(cached_disk_dist_registry):
    table = BenchmarksRoutingTable({"test_provider": BenchmarksImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple benchmarks and verify listing
    await table.register_benchmark(
        benchmark_id="test-benchmark",
        dataset_id="test-dataset",
        scoring_functions=["test-scoring-fn", "test-scoring-fn-2"],
    )
    benchmarks = await table.list_benchmarks()

    assert len(benchmarks.data) == 1
    benchmark_ids = {b.identifier for b in benchmarks.data}
    assert "test-benchmark" in benchmark_ids

    # Unregister the benchmark and verify removal
    await table.unregister_benchmark(benchmark_id="test-benchmark")
    benchmarks_after = await table.list_benchmarks()
    assert len(benchmarks_after.data) == 0

    # Unregistering a non-existent benchmark should raise a clear error
    with pytest.raises(ValueError, match="Benchmark 'dummy_benchmark' not found"):
        await table.unregister_benchmark(benchmark_id="dummy_benchmark")


async def test_tool_groups_routing_table(cached_disk_dist_registry):
    table = ToolGroupsRoutingTable({"test_provider": ToolGroupsImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register multiple tool groups and verify listing
    await table.register_tool_group(
        toolgroup_id="test-toolgroup",
        provider_id="test_provider",
    )
    tool_groups = await table.list_tool_groups()

    assert len(tool_groups.data) == 1
    tool_group_ids = {tg.identifier for tg in tool_groups.data}
    assert "test-toolgroup" in tool_group_ids

    await table.unregister_toolgroup(toolgroup_id="test-toolgroup")
    tool_groups = await table.list_tool_groups()
    assert len(tool_groups.data) == 0


async def test_models_alias_registration_and_lookup(cached_disk_dist_registry):
    """Test alias registration (model_id != provider_model_id) and lookup behavior."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model with alias (model_id different from provider_model_id)
    # NOTE: Aliases are not supported anymore, so this is a no-op
    await table.register_model(
        model_id="my-alias", provider_model_id="actual-provider-model", provider_id="test_provider"
    )

    # Verify the model was registered with alias as identifier (not namespaced)
    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.identifier == "test_provider/actual-provider-model"
    assert model.provider_resource_id == "actual-provider-model"

    # Test lookup by alias fails
    with pytest.raises(ModelNotFoundError, match="Model 'my-alias' not found"):
        await table.get_model("my-alias")

    retrieved_model = await table.get_model("test_provider/actual-provider-model")
    assert retrieved_model.identifier == "test_provider/actual-provider-model"
    assert retrieved_model.provider_resource_id == "actual-provider-model"


async def test_models_multi_provider_disambiguation(cached_disk_dist_registry):
    """Test registration and lookup with multiple providers having same provider_model_id."""
    table = ModelsRoutingTable(
        {"provider1": InferenceImpl(), "provider2": InferenceImpl()}, cached_disk_dist_registry, {}
    )
    await table.initialize()

    # Register same provider_model_id on both providers (no aliases)
    await table.register_model(model_id="common-model", provider_id="provider1")
    await table.register_model(model_id="common-model", provider_id="provider2")

    # Verify both models get namespaced identifiers
    models = await table.list_models()
    assert len(models.data) == 2
    identifiers = {m.identifier for m in models.data}
    assert identifiers == {"provider1/common-model", "provider2/common-model"}

    # Test lookup by full namespaced identifier works
    model1 = await table.get_model("provider1/common-model")
    assert model1.provider_id == "provider1"
    assert model1.provider_resource_id == "common-model"

    model2 = await table.get_model("provider2/common-model")
    assert model2.provider_id == "provider2"
    assert model2.provider_resource_id == "common-model"

    # Test lookup by unscoped provider_model_id fails with multiple providers error
    with pytest.raises(ModelNotFoundError, match="Model 'common-model' not found"):
        await table.get_model("common-model")


async def test_models_fallback_lookup_behavior(cached_disk_dist_registry):
    """Test two-stage lookup: direct identifier hit vs fallback to provider_resource_id."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model without alias (gets namespaced identifier)
    await table.register_model(model_id="test-model", provider_id="test_provider")

    # Verify namespaced identifier was created
    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.identifier == "test_provider/test-model"
    assert model.provider_resource_id == "test-model"

    # Test lookup by full namespaced identifier (direct hit via get_object_by_identifier)
    retrieved_model = await table.get_model("test_provider/test-model")
    assert retrieved_model.identifier == "test_provider/test-model"

    # Test lookup by unscoped provider_model_id (fallback via iteration)
    with pytest.raises(ModelNotFoundError, match="Model 'test-model' not found"):
        await table.get_model("test-model")

    # Test lookup of non-existent model fails
    with pytest.raises(ModelNotFoundError, match="Model 'non-existent' not found"):
        await table.get_model("non-existent")


async def test_models_source_tracking_default(cached_disk_dist_registry):
    """Test that models registered via register_model get default source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register model via register_model (should get default source)
    await table.register_model(model_id="user-model", provider_id="test_provider")

    models = await table.list_models()
    assert len(models.data) == 1
    model = models.data[0]
    assert model.source == RegistryEntrySource.via_register_api
    assert model.identifier == "test_provider/user-model"

    # Cleanup
    await table.shutdown()


async def test_models_source_tracking_provider(cached_disk_dist_registry):
    """Test that models registered via update_registered_models get provider source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Simulate provider refresh by calling update_registered_models
    provider_models = [
        Model(
            identifier="provider-model-1",
            provider_resource_id="provider-model-1",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
        Model(
            identifier="provider-model-2",
            provider_resource_id="provider-model-2",
            provider_id="test_provider",
            metadata={"embedding_dimension": 512},
            model_type=ModelType.embedding,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models)

    models = await table.list_models()
    assert len(models.data) == 2

    # All models should have provider source
    for model in models.data:
        assert model.source == RegistryEntrySource.listed_from_provider
        assert model.provider_id == "test_provider"

    # Cleanup
    await table.shutdown()


async def test_models_source_interaction_preserves_default(cached_disk_dist_registry):
    """Test that provider refresh preserves user-registered models with default source."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # First register a user model with same provider_resource_id as provider will later provide
    await table.register_model(
        model_id="my-custom-alias", provider_model_id="provider-model-1", provider_id="test_provider"
    )

    # Verify user model is registered with default source
    models = await table.list_models()
    assert len(models.data) == 1
    user_model = models.data[0]
    assert user_model.source == RegistryEntrySource.via_register_api
    assert user_model.identifier == "test_provider/provider-model-1"
    assert user_model.provider_resource_id == "provider-model-1"

    # Now simulate provider refresh
    provider_models = [
        Model(
            identifier="provider-model-1",
            provider_resource_id="provider-model-1",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
        Model(
            identifier="different-model",
            provider_resource_id="different-model",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models)

    # Verify user model with alias is preserved, but provider added new model
    models = await table.list_models()
    assert len(models.data) == 2

    # Find the user model and provider model
    user_model = next((m for m in models.data if m.identifier == "test_provider/provider-model-1"), None)
    provider_model = next((m for m in models.data if m.identifier == "test_provider/different-model"), None)

    assert user_model is not None
    assert user_model.source == RegistryEntrySource.via_register_api
    assert user_model.provider_resource_id == "provider-model-1"

    assert provider_model is not None
    assert provider_model.source == RegistryEntrySource.listed_from_provider
    assert provider_model.provider_resource_id == "different-model"

    # Cleanup
    await table.shutdown()


async def test_models_source_interaction_cleanup_provider_models(cached_disk_dist_registry):
    """Test that provider refresh removes old provider models but keeps default ones."""
    table = ModelsRoutingTable({"test_provider": InferenceImpl()}, cached_disk_dist_registry, {})
    await table.initialize()

    # Register a user model
    await table.register_model(model_id="user-model", provider_id="test_provider")

    # Add some provider models
    provider_models_v1 = [
        Model(
            identifier="provider-model-old",
            provider_resource_id="provider-model-old",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models_v1)

    # Verify we have both user and provider models
    models = await table.list_models()
    assert len(models.data) == 2

    # Now update with new provider models (should remove old provider models)
    provider_models_v2 = [
        Model(
            identifier="provider-model-new",
            provider_resource_id="provider-model-new",
            provider_id="test_provider",
            metadata={},
            model_type=ModelType.llm,
        ),
    ]
    await table.update_registered_models("test_provider", provider_models_v2)

    # Should have user model + new provider model, old provider model gone
    models = await table.list_models()
    assert len(models.data) == 2

    identifiers = {m.identifier for m in models.data}
    assert "test_provider/user-model" in identifiers  # User model preserved
    assert "test_provider/provider-model-new" in identifiers  # New provider model (uses provider's identifier)
    assert "test_provider/provider-model-old" not in identifiers  # Old provider model removed

    # Verify sources are correct
    user_model = next((m for m in models.data if m.identifier == "test_provider/user-model"), None)
    provider_model = next((m for m in models.data if m.identifier == "test_provider/provider-model-new"), None)

    assert user_model.source == RegistryEntrySource.via_register_api
    assert provider_model.source == RegistryEntrySource.listed_from_provider

    # Cleanup
    await table.shutdown()


async def test_tool_groups_routing_table_exception_handling(cached_disk_dist_registry):
    """Test that the tool group routing table handles exceptions when listing tools, like if an MCP server is unreachable."""

    exception_throwing_tool_groups_impl = ToolGroupsImpl()
    exception_throwing_tool_groups_impl.list_runtime_tools = AsyncMock(side_effect=Exception("Test exception"))

    table = ToolGroupsRoutingTable(
        {"test_provider": exception_throwing_tool_groups_impl}, cached_disk_dist_registry, {}
    )
    await table.initialize()

    await table.register_tool_group(
        toolgroup_id="test-toolgroup-exceptions",
        provider_id="test_provider",
        mcp_endpoint=URL(uri="http://localhost:8479/foo/bar"),
    )

    tools = await table.list_tools(toolgroup_id="test-toolgroup-exceptions")

    assert len(tools.data) == 0
