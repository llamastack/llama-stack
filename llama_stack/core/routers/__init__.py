# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.core.datatypes import AccessRule, RoutedProtocol
from llama_stack.core.stack import StackRunConfig
from llama_stack.core.store import DistributionRegistry
from llama_stack.providers.datatypes import Api, RoutingTable
from llama_stack.providers.utils.inference.inference_store import InferenceStore


async def get_routing_table_impl(
    api: Api,
    impls_by_provider_id: dict[str, RoutedProtocol],
    _deps,
    dist_registry: DistributionRegistry,
    policy: list[AccessRule],
) -> Any:
    from ..routing_tables.benchmarks import BenchmarksRoutingTable
    from ..routing_tables.datasets import DatasetsRoutingTable
    from ..routing_tables.models import ModelsRoutingTable
    from ..routing_tables.scoring_functions import ScoringFunctionsRoutingTable
    from ..routing_tables.shields import ShieldsRoutingTable
    from ..routing_tables.toolgroups import ToolGroupsRoutingTable

    api_to_tables = {
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
        "datasets": DatasetsRoutingTable,
        "scoring_functions": ScoringFunctionsRoutingTable,
        "benchmarks": BenchmarksRoutingTable,
        "tool_groups": ToolGroupsRoutingTable,
    }

    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](impls_by_provider_id, dist_registry, policy)
    await impl.initialize()
    return impl


async def get_auto_router_impl(
    api: Api, routing_table: RoutingTable, deps: dict[str, Any], run_config: StackRunConfig, policy: list[AccessRule]
) -> Any:
    from .datasets import DatasetIORouter
    from .eval_scoring import EvalRouter, ScoringRouter
    from .inference import InferenceRouter
    from .safety import SafetyRouter
    from .tool_runtime import ToolRuntimeRouter
    from .vector_io import VectorIORouter

    api_to_routers = {
        "vector_io": VectorIORouter,
        "inference": InferenceRouter,
        "safety": SafetyRouter,
        "datasetio": DatasetIORouter,
        "scoring": ScoringRouter,
        "eval": EvalRouter,
        "tool_runtime": ToolRuntimeRouter,
    }
    api_to_deps = {
        "inference": {"telemetry": Api.telemetry},
    }
    if api.value not in api_to_routers:
        raise ValueError(f"API {api.value} not found in router map")

    api_to_dep_impl = {}
    for dep_name, dep_api in api_to_deps.get(api.value, {}).items():
        if dep_api in deps:
            api_to_dep_impl[dep_name] = deps[dep_api]

    # TODO: move pass configs to routers instead
    if api == Api.inference and run_config.inference_store:
        inference_store = InferenceStore(
            config=run_config.inference_store,
            policy=policy,
        )
        await inference_store.initialize()
        api_to_dep_impl["store"] = inference_store

    impl = api_to_routers[api.value](routing_table, **api_to_dep_impl)
    await impl.initialize()
    return impl
