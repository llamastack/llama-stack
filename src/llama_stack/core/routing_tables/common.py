# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.errors import ModelNotFoundError
from llama_stack.apis.models import Model
from llama_stack.apis.resource import ResourceType
from llama_stack.core.access_control.access_control import AccessDeniedError, is_action_allowed
from llama_stack.core.access_control.datatypes import Action
from llama_stack.core.datatypes import (
    AccessRule,
    RoutableObject,
    RoutableObjectWithProvider,
    RoutedProtocol,
    ScoringFnWithOwner,
)
from llama_stack.core.request_headers import get_authenticated_user
from llama_stack.core.store import DistributionRegistry
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api, RoutingTable

logger = get_logger(name=__name__, category="core::routing_tables")


def get_impl_api(p: Any) -> Api:
    return p.__provider_spec__.api


# TODO: this should return the registered object for all APIs
async def register_object_with_provider(obj: RoutableObject, p: Any) -> RoutableObject:
    api = get_impl_api(p)

    assert obj.provider_id != "remote", "Remote provider should not be registered"

    if api == Api.inference:
        return await p.register_model(obj)
    elif api == Api.safety:
        return await p.register_shield(obj)
    elif api == Api.vector_io:
        return await p.register_vector_store(obj)
    elif api == Api.datasetio:
        return await p.register_dataset(obj)
    elif api == Api.scoring:
        return await p.register_scoring_function(obj)
    elif api == Api.eval:
        return await p.register_benchmark(obj)
    elif api == Api.tool_runtime:
        return await p.register_toolgroup(obj)
    else:
        raise ValueError(f"Unknown API {api} for registering object with provider")


async def unregister_object_from_provider(obj: RoutableObject, p: Any) -> None:
    api = get_impl_api(p)
    if api == Api.vector_io:
        return await p.unregister_vector_store(obj.identifier)
    elif api == Api.inference:
        return await p.unregister_model(obj.identifier)
    elif api == Api.safety:
        return await p.unregister_shield(obj.identifier)
    elif api == Api.datasetio:
        return await p.unregister_dataset(obj.identifier)
    elif api == Api.eval:
        return await p.unregister_benchmark(obj.identifier)
    elif api == Api.scoring:
        return await p.unregister_scoring_function(obj.identifier)
    elif api == Api.tool_runtime:
        return await p.unregister_toolgroup(obj.identifier)
    else:
        raise ValueError(f"Unregister not supported for {api}")


Registry = dict[str, list[RoutableObjectWithProvider]]


class CommonRoutingTableImpl(RoutingTable):
    def __init__(
        self,
        impls_by_provider_id: dict[str, RoutedProtocol],
        dist_registry: DistributionRegistry,
        policy: list[AccessRule],
    ) -> None:
        self.impls_by_provider_id = impls_by_provider_id
        self.dist_registry = dist_registry
        self.policy = policy

    async def initialize(self) -> None:
        async def add_objects(objs: list[RoutableObjectWithProvider], provider_id: str, cls) -> None:
            for obj in objs:
                if cls is None:
                    obj.provider_id = provider_id
                else:
                    # Create a copy of the model data and explicitly set provider_id
                    model_data = obj.model_dump()
                    model_data["provider_id"] = provider_id
                    obj = cls(**model_data)
                await self.dist_registry.register(obj)

        # Register all objects from providers
        for pid, p in self.impls_by_provider_id.items():
            api = get_impl_api(p)
            if api == Api.inference:
                p.model_store = self
            elif api == Api.safety:
                p.shield_store = self
            elif api == Api.vector_io:
                p.vector_store_store = self
            elif api == Api.datasetio:
                p.dataset_store = self
            elif api == Api.scoring:
                p.scoring_function_store = self
                scoring_functions = await p.list_scoring_functions()
                await add_objects(scoring_functions, pid, ScoringFnWithOwner)
            elif api == Api.eval:
                p.benchmark_store = self
            elif api == Api.tool_runtime:
                p.tool_store = self

    async def shutdown(self) -> None:
        for p in self.impls_by_provider_id.values():
            await p.shutdown()

    async def refresh(self) -> None:
        pass

    async def get_provider_impl(self, routing_key: str, provider_id: str | None = None) -> Any:
        from .benchmarks import BenchmarksRoutingTable
        from .datasets import DatasetsRoutingTable
        from .models import ModelsRoutingTable
        from .scoring_functions import ScoringFunctionsRoutingTable
        from .shields import ShieldsRoutingTable
        from .toolgroups import ToolGroupsRoutingTable
        from .vector_stores import VectorStoresRoutingTable

        def apiname_object():
            if isinstance(self, ModelsRoutingTable):
                return ("Inference", "model")
            elif isinstance(self, ShieldsRoutingTable):
                return ("Safety", "shield")
            elif isinstance(self, VectorStoresRoutingTable):
                return ("VectorIO", "vector_store")
            elif isinstance(self, DatasetsRoutingTable):
                return ("DatasetIO", "dataset")
            elif isinstance(self, ScoringFunctionsRoutingTable):
                return ("Scoring", "scoring_function")
            elif isinstance(self, BenchmarksRoutingTable):
                return ("Eval", "benchmark")
            elif isinstance(self, ToolGroupsRoutingTable):
                return ("ToolGroups", "tool_group")
            else:
                raise ValueError("Unknown routing table type")

        apiname, objtype = apiname_object()

        # Get objects from disk registry
        obj = self.dist_registry.get_cached(objtype, routing_key)
        if not obj:
            provider_ids = list(self.impls_by_provider_id.keys())
            if len(provider_ids) > 1:
                provider_ids_str = f"any of the providers: {', '.join(provider_ids)}"
            else:
                provider_ids_str = f"provider: `{provider_ids[0]}`"
            raise ValueError(
                f"{objtype.capitalize()} `{routing_key}` not served by {provider_ids_str}. Make sure there is an {apiname} provider serving this {objtype}."
            )

        if not provider_id or provider_id == obj.provider_id:
            return self.impls_by_provider_id[obj.provider_id]

        raise ValueError(f"Provider not found for `{routing_key}`")

    async def get_object_by_identifier(self, type: str, identifier: str) -> RoutableObjectWithProvider | None:
        # Get from disk registry
        obj = await self.dist_registry.get(type, identifier)
        if not obj:
            return None

        # Check if user has permission to access this object
        if not is_action_allowed(self.policy, "read", obj, get_authenticated_user()):
            logger.debug(f"Access denied to {type} '{identifier}'")
            return None

        return obj

    async def unregister_object(self, obj: RoutableObjectWithProvider) -> None:
        user = get_authenticated_user()
        if not is_action_allowed(self.policy, "delete", obj, user):
            raise AccessDeniedError("delete", obj, user)
        await self.dist_registry.delete(obj.type, obj.identifier)
        await unregister_object_from_provider(obj, self.impls_by_provider_id[obj.provider_id])

    async def register_object(self, obj: RoutableObjectWithProvider) -> RoutableObjectWithProvider:
        # if provider_id is not specified, pick an arbitrary one from existing entries
        if not obj.provider_id and len(self.impls_by_provider_id) > 0:
            obj.provider_id = list(self.impls_by_provider_id.keys())[0]

        if obj.provider_id not in self.impls_by_provider_id:
            raise ValueError(f"Provider `{obj.provider_id}` not found")

        p = self.impls_by_provider_id[obj.provider_id]

        # If object supports access control but no attributes set, use creator's attributes
        creator = get_authenticated_user()
        if not is_action_allowed(self.policy, "create", obj, creator):
            raise AccessDeniedError("create", obj, creator)
        if creator:
            obj.owner = creator
            logger.info(f"Setting owner for {obj.type} '{obj.identifier}' to {obj.owner.principal}")

        registered_obj = await register_object_with_provider(obj, p)
        # TODO: This needs to be fixed for all APIs once they return the registered object
        if obj.type == ResourceType.model.value:
            await self.dist_registry.register(registered_obj)
            return registered_obj
        else:
            await self.dist_registry.register(obj)
            return obj

    async def assert_action_allowed(
        self,
        action: Action,
        type: str,
        identifier: str,
    ) -> None:
        """Fetch a registered object by type/identifier and enforce the given action permission."""
        obj = await self.get_object_by_identifier(type, identifier)
        if obj is None:
            raise ValueError(f"{type.capitalize()} '{identifier}' not found")
        user = get_authenticated_user()
        if not is_action_allowed(self.policy, action, obj, user):
            raise AccessDeniedError(action, obj, user)

    async def get_all_with_type(self, type: str) -> list[RoutableObjectWithProvider]:
        objs = await self.dist_registry.get_all()
        filtered_objs = [obj for obj in objs if obj.type == type]

        # Apply attribute-based access control filtering
        if filtered_objs:
            filtered_objs = [
                obj for obj in filtered_objs if is_action_allowed(self.policy, "read", obj, get_authenticated_user())
            ]

        return filtered_objs


async def lookup_model(routing_table: CommonRoutingTableImpl, model_id: str) -> Model:
    model = await routing_table.get_object_by_identifier("model", model_id)
    if not model:
        raise ModelNotFoundError(model_id)
    return model
