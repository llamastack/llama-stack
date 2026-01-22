# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import importlib.resources
import inspect
import os
import re
import tempfile
from typing import Any, get_type_hints

import yaml
from pydantic import BaseModel

from llama_stack.core.admin import AdminImpl, AdminImplConfig
from llama_stack.core.conversations.conversations import ConversationServiceConfig, ConversationServiceImpl
from llama_stack.core.datatypes import Provider, QualifiedModel, SafetyConfig, StackConfig, VectorStoresConfig
from llama_stack.core.distribution import get_provider_registry
from llama_stack.core.inspect import DistributionInspectConfig, DistributionInspectImpl
from llama_stack.core.prompts.prompts import PromptServiceConfig, PromptServiceImpl
from llama_stack.core.providers import ProviderImpl, ProviderImplConfig
from llama_stack.core.resolver import ProviderRegistry, resolve_impls
from llama_stack.core.routing_tables.common import CommonRoutingTableImpl
from llama_stack.core.storage.datatypes import (
    InferenceStoreReference,
    KVStoreReference,
    ServerStoresConfig,
    SqliteKVStoreConfig,
    SqliteSqlStoreConfig,
    SqlStoreReference,
    StorageBackendConfig,
    StorageConfig,
)
from llama_stack.core.store.registry import create_dist_registry
from llama_stack.core.utils.dynamic import instantiate_class_type
from llama_stack.log import get_logger
from llama_stack_api import (
    Agents,
    Api,
    Batches,
    Benchmarks,
    Conversations,
    DatasetIO,
    Datasets,
    Eval,
    Files,
    Inference,
    Inspect,
    Models,
    PostTraining,
    Prompts,
    Providers,
    RegisterBenchmarkRequest,
    Safety,
    Scoring,
    ScoringFunctions,
    Shields,
    ToolGroups,
    ToolRuntime,
    VectorIO,
)
from llama_stack_api.datasets import RegisterDatasetRequest

logger = get_logger(name=__name__, category="core")


class LlamaStack(
    Providers,
    Inference,
    Agents,
    Batches,
    Safety,
    Datasets,
    PostTraining,
    VectorIO,
    Eval,
    Benchmarks,
    Scoring,
    ScoringFunctions,
    DatasetIO,
    Models,
    Shields,
    Inspect,
    ToolGroups,
    ToolRuntime,
    Files,
    Prompts,
    Conversations,
):
    pass


# Resources to register based on configuration.
# If a request class is specified, the configuration object will be converted to this class before invoking the registration method.
RESOURCES = [
    ("models", Api.models, "register_model", "list_models", None),
    ("shields", Api.shields, "register_shield", "list_shields", None),
    ("datasets", Api.datasets, "register_dataset", "list_datasets", RegisterDatasetRequest),
    (
        "scoring_fns",
        Api.scoring_functions,
        "register_scoring_function",
        "list_scoring_functions",
        None,
    ),
    ("benchmarks", Api.benchmarks, "register_benchmark", "list_benchmarks", RegisterBenchmarkRequest),
    ("tool_groups", Api.tool_groups, "register_tool_group", "list_tool_groups", None),
    ("vector_stores", Api.vector_stores, "register_vector_store", "list_vector_stores", None),
]


REGISTRY_REFRESH_INTERVAL_SECONDS = 300
REGISTRY_REFRESH_TASK = None
TEST_RECORDING_CONTEXT = None

# ID fields for registered resources that should trigger skipping
# when they resolve to empty/None (from conditional env vars like :+)
RESOURCE_ID_FIELDS = [
    "vector_store_id",
    "model_id",
    "shield_id",
    "dataset_id",
    "scoring_fn_id",
    "benchmark_id",
    "toolgroup_id",
]


def is_request_model(t: Any) -> bool:
    """Check if a type is a request model (Pydantic BaseModel).

    Args:
        t: The type to check

    Returns:
        True if the type is a Pydantic BaseModel subclass, False otherwise
    """

    return inspect.isclass(t) and issubclass(t, BaseModel)


async def invoke_with_optional_request(method: Any) -> Any:
    """Invoke a method, automatically creating a request instance if needed.

    For APIs that use request models, this will create an empty request object.
    For backward compatibility, falls back to calling without arguments.

    Uses get_type_hints() to resolve forward references (e.g., "ListBenchmarksRequest" -> actual class).

    Handles methods with:
    - No parameters: calls without arguments
    - One or more request model parameters: creates empty instances for each
    - Mixed parameters: creates request models, uses defaults for others
    - Required non-request-model parameters without defaults: falls back to calling without arguments

    Args:
        method: The method to invoke

    Returns:
        The result of calling the method
    """
    try:
        hints = get_type_hints(method)
    except Exception:
        # Forward references can't be resolved, fall back to calling without request
        return await method()

    params = list(inspect.signature(method).parameters.values())
    params = [p for p in params if p.name != "self"]

    if not params:
        return await method()

    # Build arguments for the method call
    args: dict[str, Any] = {}
    can_call = True

    for param in params:
        param_type = hints.get(param.name)

        # If it's a request model, try to create an empty instance
        if param_type and is_request_model(param_type):
            try:
                args[param.name] = param_type()
            except Exception:
                # Request model requires arguments, can't create empty instance
                can_call = False
                break
        # If it has a default value, we can skip it (will use default)
        elif param.default != inspect.Parameter.empty:
            continue
        # Required parameter that's not a request model - can't provide it
        else:
            can_call = False
            break

    if can_call and args:
        return await method(**args)

    # Fall back to calling without arguments for backward compatibility
    return await method()


async def register_resources(run_config: StackConfig, impls: dict[Api, Any]):
    for rsrc, api, register_method, list_method, request_class in RESOURCES:
        objects = getattr(run_config.registered_resources, rsrc)
        if api not in impls:
            continue

        method = getattr(impls[api], register_method)
        for obj in objects:
            if hasattr(obj, "provider_id"):
                # Do not register models on disabled providers
                if not obj.provider_id or obj.provider_id == "__disabled__":
                    logger.debug(f"Skipping {rsrc.capitalize()} registration for disabled provider.")
                    continue
                logger.debug(f"registering {rsrc.capitalize()} {obj} for provider {obj.provider_id}")

            # TODO: Once all register methods are migrated to accept request objects,
            # remove this conditional and always use the request_class pattern.
            if request_class is not None:
                request = request_class(**obj.model_dump())
                await method(request)
            else:
                # we want to maintain the type information in arguments to method.
                # instead of method(**obj.model_dump()), which may convert a typed attr to a dict,
                # we use model_dump() to find all the attrs and then getattr to get the still typed
                # value.
                await method(**{k: getattr(obj, k) for k in obj.model_dump().keys()})

        method = getattr(impls[api], list_method)
        response = await invoke_with_optional_request(method)

        objects_to_process = response.data if hasattr(response, "data") else response

        for obj in objects_to_process:
            logger.debug(
                f"{rsrc.capitalize()}: {obj.identifier} served by {obj.provider_id}",
            )


async def validate_vector_stores_config(vector_stores_config: VectorStoresConfig | None, impls: dict[Api, Any]):
    """Validate vector stores configuration."""
    if vector_stores_config is None:
        return

    # Validate default embedding model
    if vector_stores_config.default_embedding_model is not None:
        await _validate_embedding_model(vector_stores_config.default_embedding_model, impls)

    # Validate rewrite query params
    if vector_stores_config.rewrite_query_params:
        if vector_stores_config.rewrite_query_params.model:
            await _validate_rewrite_query_model(vector_stores_config.rewrite_query_params.model, impls)


async def _validate_embedding_model(embedding_model: QualifiedModel, impls: dict[Api, Any]) -> None:
    """Validate that an embedding model exists and has required metadata."""
    provider_id = embedding_model.provider_id
    model_id = embedding_model.model_id
    model_identifier = f"{provider_id}/{model_id}"

    if Api.models not in impls:
        raise ValueError(f"Models API is not available but vector_stores config requires model '{model_identifier}'")

    models_impl = impls[Api.models]
    response = await models_impl.list_models()
    models_list = {m.identifier: m for m in response.data if m.model_type == "embedding"}

    model = models_list.get(model_identifier)
    if model is None:
        raise ValueError(
            f"Embedding model '{model_identifier}' not found. Available embedding models: {list(models_list.keys())}"
        )

    embedding_dimension = model.metadata.get("embedding_dimension")
    if embedding_dimension is None:
        raise ValueError(f"Embedding model '{model_identifier}' is missing 'embedding_dimension' in metadata")

    try:
        int(embedding_dimension)
    except ValueError as err:
        raise ValueError(f"Embedding dimension '{embedding_dimension}' cannot be converted to an integer") from err

    logger.debug(f"Validated embedding model: {model_identifier} (dimension: {embedding_dimension})")


async def _validate_rewrite_query_model(rewrite_query_model: QualifiedModel, impls: dict[Api, Any]) -> None:
    """Validate that a rewrite query model exists and is accessible."""
    provider_id = rewrite_query_model.provider_id
    model_id = rewrite_query_model.model_id
    model_identifier = f"{provider_id}/{model_id}"

    if Api.models not in impls:
        raise ValueError(
            f"Models API is not available but vector_stores config requires rewrite query model '{model_identifier}'"
        )

    models_impl = impls[Api.models]
    response = await models_impl.list_models()
    llm_models_list = {m.identifier: m for m in response.data if m.model_type == "llm"}

    model = llm_models_list.get(model_identifier)
    if model is None:
        raise ValueError(
            f"Rewrite query model '{model_identifier}' not found. Available LLM models: {list(llm_models_list.keys())}"
        )

    logger.debug(f"Validated rewrite query model: {model_identifier}")


async def validate_safety_config(safety_config: SafetyConfig | None, impls: dict[Api, Any]):
    if safety_config is None or safety_config.default_shield_id is None:
        return

    if Api.shields not in impls:
        raise ValueError("Safety configuration requires the shields API to be enabled")

    if Api.safety not in impls:
        raise ValueError("Safety configuration requires the safety API to be enabled")

    shields_impl = impls[Api.shields]
    response = await shields_impl.list_shields()
    shields_by_id = {shield.identifier: shield for shield in response.data}

    default_shield_id = safety_config.default_shield_id
    # don't validate if there are no shields registered
    if shields_by_id and default_shield_id not in shields_by_id:
        available = sorted(shields_by_id)
        raise ValueError(
            f"Configured default_shield_id '{default_shield_id}' not found among registered shields."
            f" Available shields: {available}"
        )


class EnvVarError(Exception):
    def __init__(self, var_name: str, path: str = ""):
        self.var_name = var_name
        self.path = path
        super().__init__(
            f"Environment variable '{var_name}' not set or empty {f'at {path}' if path else ''}. "
            f"Use ${{env.{var_name}:=default_value}} to provide a default value, "
            f"${{env.{var_name}:+value_if_set}} to make the field conditional, "
            f"or ensure the environment variable is set."
        )


def replace_env_vars(config: Any, path: str = "") -> Any:
    if isinstance(config, dict):
        result = {}
        for k, v in config.items():
            try:
                result[k] = replace_env_vars(v, f"{path}.{k}" if path else k)
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, list):
        result = []
        for i, v in enumerate(config):
            try:
                # Special handling for providers: first resolve the provider_id to check if provider
                # is disabled so that we can skip config env variable expansion and avoid validation errors
                if isinstance(v, dict) and "provider_id" in v:
                    try:
                        resolved_provider_id = replace_env_vars(v["provider_id"], f"{path}[{i}].provider_id")
                        if resolved_provider_id == "__disabled__":
                            logger.debug(
                                f"Skipping config env variable expansion for disabled provider: {v.get('provider_id', '')}"
                            )
                            continue
                    except EnvVarError:
                        # If we can't resolve the provider_id, continue with normal processing
                        pass

                # Special handling for registered resources: check if ID field resolves to empty/None
                # from conditional env vars (e.g., ${env.VAR:+value}) and skip the entry if so
                if isinstance(v, dict):
                    should_skip = False
                    for id_field in RESOURCE_ID_FIELDS:
                        if id_field in v:
                            try:
                                resolved_id = replace_env_vars(v[id_field], f"{path}[{i}].{id_field}")
                                if resolved_id is None or resolved_id == "":
                                    logger.debug(
                                        f"Skipping {path}[{i}] with empty {id_field} (conditional env var not set)"
                                    )
                                    should_skip = True
                                    break
                            except EnvVarError as e:
                                logger.warning(
                                    f"Could not resolve {id_field} in {path}[{i}], env var '{e.var_name}': {e}"
                                )
                    if should_skip:
                        continue

                # Normal processing
                result.append(replace_env_vars(v, f"{path}[{i}]"))
            except EnvVarError as e:
                raise EnvVarError(e.var_name, e.path) from None
        return result

    elif isinstance(config, str):
        # Pattern supports bash-like syntax: := for default and :+ for conditional and a optional value
        pattern = r"\${env\.([A-Z0-9_]+)(?::([=+])([^}]*))?}"

        def get_env_var(match: re.Match):
            env_var = match.group(1)
            operator = match.group(2)  # '=' for default, '+' for conditional
            value_expr = match.group(3)

            env_value = os.environ.get(env_var)

            if operator == "=":  # Default value syntax: ${env.FOO:=default}
                # If the env is set like ${env.FOO:=default} then use the env value when set
                if env_value:
                    value = env_value
                else:
                    # If the env is not set, look for a default value
                    # value_expr returns empty string (not None) when not matched
                    # This means ${env.FOO:=} and it's accepted and returns empty string - just like bash
                    if value_expr == "":
                        return ""
                    else:
                        value = value_expr

            elif operator == "+":  # Conditional value syntax: ${env.FOO:+value_if_set}
                # If the env is set like ${env.FOO:+value_if_set} then use the value_if_set
                if env_value:
                    if value_expr:
                        value = value_expr
                    # This means ${env.FOO:+}
                    else:
                        # Just like bash, this doesn't care whether the env is set or not and applies
                        # the value, in this case the empty string
                        return ""
                else:
                    # Just like bash, this doesn't care whether the env is set or not, since it's not set
                    # we return an empty string
                    value = ""
            else:  # No operator case: ${env.FOO}
                if not env_value:
                    raise EnvVarError(env_var, path)
                value = env_value

            # expand "~" from the values
            return os.path.expanduser(value)

        try:
            result = re.sub(pattern, get_env_var, config)
            # Only apply type conversion if substitution actually happened
            if result != config:
                return _convert_string_to_proper_type(result)
            return result
        except EnvVarError as e:
            raise EnvVarError(e.var_name, e.path) from None

    return config


def _convert_string_to_proper_type(value: str) -> Any:
    # This might be tricky depending on what the config type is, if  'str | None' we are
    # good, if 'str' we need to keep the empty string... 'str | None' is more common and
    # providers config should be typed this way.
    # TODO: we could try to load the config class and see if the config has a field with type 'str | None'
    # and then convert the empty string to None or not
    if value == "":
        return None

    lowered = value.lower()
    if lowered == "true":
        return True
    elif lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def cast_image_name_to_string(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensure that any value for a key 'image_name' in a config_dict is a string"""
    if "image_name" in config_dict and config_dict["image_name"] is not None:
        config_dict["image_name"] = str(config_dict["image_name"])
    return config_dict


def add_internal_implementations(impls: dict[Api, Any], config: StackConfig) -> None:
    """Add internal implementations (inspect, providers, and admin) to the implementations dictionary.
    Args:
        impls: Dictionary of API implementations
        run_config: Stack run configuration
    """
    inspect_impl = DistributionInspectImpl(
        DistributionInspectConfig(config=config),
        deps=impls,
    )
    impls[Api.inspect] = inspect_impl

    providers_impl = ProviderImpl(
        ProviderImplConfig(config=config),
        deps=impls,
    )
    impls[Api.providers] = providers_impl

    admin_impl = AdminImpl(
        AdminImplConfig(config=config),
        deps=impls,
    )
    impls[Api.admin] = admin_impl

    prompts_impl = PromptServiceImpl(
        PromptServiceConfig(config=config),
        deps=impls,
    )
    impls[Api.prompts] = prompts_impl

    conversations_impl = ConversationServiceImpl(
        ConversationServiceConfig(config=config),
        deps=impls,
    )
    impls[Api.conversations] = conversations_impl


def _initialize_storage(run_config: StackConfig):
    kv_backends: dict[str, StorageBackendConfig] = {}
    sql_backends: dict[str, StorageBackendConfig] = {}
    for backend_name, backend_config in run_config.storage.backends.items():
        type = backend_config.type.value
        if type.startswith("kv_"):
            kv_backends[backend_name] = backend_config
        elif type.startswith("sql_"):
            sql_backends[backend_name] = backend_config
        else:
            raise ValueError(f"Unknown storage backend type: {type}")

    from llama_stack.core.storage.kvstore.kvstore import register_kvstore_backends
    from llama_stack.core.storage.sqlstore.sqlstore import register_sqlstore_backends

    register_kvstore_backends(kv_backends)
    register_sqlstore_backends(sql_backends)


class Stack:
    def __init__(self, run_config: StackConfig, provider_registry: ProviderRegistry | None = None):
        self.run_config = run_config
        self.provider_registry = provider_registry
        self.impls = None

    # Produces a stack of providers for the given run config. Not all APIs may be
    # asked for in the run config.
    async def initialize(self):
        if "LLAMA_STACK_TEST_INFERENCE_MODE" in os.environ:
            from llama_stack.testing.api_recorder import setup_api_recording

            global TEST_RECORDING_CONTEXT
            TEST_RECORDING_CONTEXT = setup_api_recording()
            if TEST_RECORDING_CONTEXT:
                TEST_RECORDING_CONTEXT.__enter__()
                logger.info(f"API recording enabled: mode={os.environ.get('LLAMA_STACK_TEST_INFERENCE_MODE')}")

        _initialize_storage(self.run_config)
        stores = self.run_config.storage.stores
        if not stores.metadata:
            raise ValueError("storage.stores.metadata must be configured with a kv_* backend")
        dist_registry, _ = await create_dist_registry(stores.metadata, self.run_config.image_name)
        policy = self.run_config.server.auth.access_policy if self.run_config.server.auth else []

        internal_impls = {}
        add_internal_implementations(internal_impls, self.run_config)

        impls = await resolve_impls(
            self.run_config,
            self.provider_registry or get_provider_registry(self.run_config),
            dist_registry,
            policy,
            internal_impls,
        )

        if Api.prompts in impls:
            await impls[Api.prompts].initialize()
        if Api.conversations in impls:
            await impls[Api.conversations].initialize()

        await register_resources(self.run_config, impls)
        await refresh_registry_once(impls)
        await validate_vector_stores_config(self.run_config.vector_stores, impls)
        await validate_safety_config(self.run_config.safety, impls)
        self.impls = impls

    def create_registry_refresh_task(self):
        assert self.impls is not None, "Must call initialize() before starting"

        global REGISTRY_REFRESH_TASK
        REGISTRY_REFRESH_TASK = asyncio.create_task(refresh_registry_task(self.impls))

        def cb(task):
            import traceback

            if task.cancelled():
                logger.error("Model refresh task cancelled")
            elif task.exception():
                logger.error(f"Model refresh task failed: {task.exception()}")
                traceback.print_exception(task.exception())
            else:
                logger.debug("Model refresh task completed")

        REGISTRY_REFRESH_TASK.add_done_callback(cb)

    async def shutdown(self):
        for impl in self.impls.values():
            impl_name = impl.__class__.__name__
            logger.debug(f"Shutting down {impl_name}")
            try:
                if hasattr(impl, "shutdown"):
                    await asyncio.wait_for(impl.shutdown(), timeout=5)
                else:
                    logger.warning(f"No shutdown method for {impl_name}")
            except TimeoutError:
                logger.exception(f"Shutdown timeout for {impl_name}")
            except (Exception, asyncio.CancelledError) as e:
                logger.exception(f"Failed to shutdown {impl_name}: {e}")

        global TEST_RECORDING_CONTEXT
        if TEST_RECORDING_CONTEXT:
            try:
                TEST_RECORDING_CONTEXT.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error during API recording cleanup: {e}")

        global REGISTRY_REFRESH_TASK
        if REGISTRY_REFRESH_TASK:
            REGISTRY_REFRESH_TASK.cancel()

        # Shutdown storage backends
        from llama_stack.core.storage.kvstore.kvstore import shutdown_kvstore_backends
        from llama_stack.core.storage.sqlstore.sqlstore import shutdown_sqlstore_backends

        try:
            await shutdown_kvstore_backends()
        except Exception as e:
            logger.exception(f"Failed to shutdown KV store backends: {e}")

        try:
            await shutdown_sqlstore_backends()
        except Exception as e:
            logger.exception(f"Failed to shutdown SQL store backends: {e}")


async def refresh_registry_once(impls: dict[Api, Any]):
    logger.debug("refreshing registry")
    routing_tables = [v for v in impls.values() if isinstance(v, CommonRoutingTableImpl)]
    for routing_table in routing_tables:
        await routing_table.refresh()


async def refresh_registry_task(impls: dict[Api, Any]):
    logger.info("starting registry refresh task")
    while True:
        await refresh_registry_once(impls)

        await asyncio.sleep(REGISTRY_REFRESH_INTERVAL_SECONDS)


def get_stack_run_config_from_distro(distro: str) -> StackConfig:
    distro_path = importlib.resources.files("llama_stack") / f"distributions/{distro}/config.yaml"

    with importlib.resources.as_file(distro_path) as path:
        if not path.exists():
            raise ValueError(f"Distribution '{distro}' not found at {distro_path}")
        run_config = yaml.safe_load(path.open())

    return StackConfig(**replace_env_vars(run_config))


def run_config_from_adhoc_config_spec(
    adhoc_config_spec: str, provider_registry: ProviderRegistry | None = None
) -> StackConfig:
    """
    Create an adhoc distribution from a list of API providers.

    The list should be of the form "api=provider", e.g. "inference=fireworks". If you have
    multiple pairs, separate them with commas or semicolons, e.g. "inference=fireworks,safety=llama-guard,agents=meta-reference"
    """

    api_providers = adhoc_config_spec.replace(";", ",").split(",")
    provider_registry = provider_registry or get_provider_registry()

    distro_dir = tempfile.mkdtemp()
    provider_configs_by_api = {}
    for api_provider in api_providers:
        api_str, provider = api_provider.split("=")
        api = Api(api_str)

        providers_by_type = provider_registry[api]
        provider_spec = providers_by_type.get(provider)
        if not provider_spec:
            provider_spec = providers_by_type.get(f"inline::{provider}")
        if not provider_spec:
            provider_spec = providers_by_type.get(f"remote::{provider}")

        if not provider_spec:
            raise ValueError(
                f"Provider {provider} (or remote::{provider} or inline::{provider}) not found for API {api}"
            )

        # call method "sample_run_config" on the provider spec config class
        provider_config_type = instantiate_class_type(provider_spec.config_class)
        provider_config = replace_env_vars(provider_config_type.sample_run_config(__distro_dir__=distro_dir))

        provider_configs_by_api[api_str] = [
            Provider(
                provider_id=provider,
                provider_type=provider_spec.provider_type,
                config=provider_config,
            )
        ]
    config = StackConfig(
        image_name="distro-test",
        apis=list(provider_configs_by_api.keys()),
        providers=provider_configs_by_api,
        storage=StorageConfig(
            backends={
                "kv_default": SqliteKVStoreConfig(db_path=f"{distro_dir}/kvstore.db"),
                "sql_default": SqliteSqlStoreConfig(db_path=f"{distro_dir}/sql_store.db"),
            },
            stores=ServerStoresConfig(
                metadata=KVStoreReference(backend="kv_default", namespace="registry"),
                inference=InferenceStoreReference(backend="sql_default", table_name="inference_store"),
                conversations=SqlStoreReference(backend="sql_default", table_name="openai_conversations"),
                prompts=KVStoreReference(backend="kv_default", namespace="prompts"),
            ),
        ),
    )
    return config
