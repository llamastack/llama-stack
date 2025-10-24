# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import glob
import importlib
import os
from typing import Any

import yaml
from pydantic import BaseModel

from llama_stack.core.datatypes import BuildConfig, DistributionSpec
from llama_stack.core.external import load_external_apis
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)

logger = get_logger(name=__name__, category="core")


INTERNAL_APIS = {Api.inspect, Api.providers, Api.prompts, Api.conversations}


def stack_apis() -> list[Api]:
    return list(Api)


class AutoRoutedApiInfo(BaseModel):
    routing_table_api: Api
    router_api: Api


def builtin_automatically_routed_apis() -> list[AutoRoutedApiInfo]:
    return [
        AutoRoutedApiInfo(
            routing_table_api=Api.models,
            router_api=Api.inference,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.shields,
            router_api=Api.safety,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.datasets,
            router_api=Api.datasetio,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.scoring_functions,
            router_api=Api.scoring,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.benchmarks,
            router_api=Api.eval,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.tool_groups,
            router_api=Api.tool_runtime,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.vector_stores,
            router_api=Api.vector_io,
        ),
    ]


def providable_apis() -> list[Api]:
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    return [api for api in Api if api not in routing_table_apis and api not in INTERNAL_APIS]


def _load_remote_provider_spec(spec_data: dict[str, Any], api: Api) -> ProviderSpec:
    spec = RemoteProviderSpec(api=api, provider_type=f"remote::{spec_data['adapter_type']}", **spec_data)
    return spec


def _load_inline_provider_spec(spec_data: dict[str, Any], api: Api, provider_name: str) -> ProviderSpec:
    spec = InlineProviderSpec(api=api, provider_type=f"inline::{provider_name}", **spec_data)
    return spec


def get_provider_registry(config=None) -> dict[Api, dict[str, ProviderSpec]]:
    """Get the provider registry, optionally including external providers.

    This function loads both built-in providers and external providers from YAML files or from their provided modules.
    External providers are loaded from a directory structure like:

    providers.d/
      remote/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml
      inline/
        inference/
          custom_ollama.yaml
          vllm.yaml
        vector_io/
          qdrant.yaml
        safety/
          llama-guard.yaml

    This method is overloaded in that it can be called from a variety of places: during build, during run, during stack construction.
    So when building external providers from a module, there are scenarios where the pip package required to import the module might not be available yet.
    There is special handling for all of the potential cases this method can be called from.

    Args:
        config: Optional object containing the external providers directory path
        building: Optional bool delineating whether or not this is being called from a build process

    Returns:
        A dictionary mapping APIs to their available providers

    Raises:
        FileNotFoundError: If the external providers directory doesn't exist
        ValueError: If any provider spec is invalid
    """

    registry: dict[Api, dict[str, ProviderSpec]] = {}
    for api in providable_apis():
        name = api.name.lower()
        logger.debug(f"Importing module {name}")
        try:
            module = importlib.import_module(f"llama_stack.providers.registry.{name}")
            registry[api] = {a.provider_type: a for a in module.available_providers()}
        except ImportError as e:
            logger.warning(f"Failed to import module {name}: {e}")

    # Refresh providable APIs with external APIs if any
    external_apis = load_external_apis(config)
    for api, api_spec in external_apis.items():
        name = api_spec.name.lower()
        logger.info(f"Importing external API {name} module {api_spec.module}")
        try:
            module = importlib.import_module(api_spec.module)
            registry[api] = {a.provider_type: a for a in module.available_providers()}
        except (ImportError, AttributeError) as e:
            # Populate the registry with an empty dict to avoid breaking the provider registry
            # This assume that the in-tree provider(s) are not available for this API which means
            # that users will need to use external providers for this API.
            registry[api] = {}
            logger.error(
                f"Failed to import external API {name}: {e}. Could not populate the in-tree provider(s) registry for {api.name}. \n"
                "Install the API package to load any in-tree providers for this API."
            )

    # Check if config has external providers
    if config:
        if hasattr(config, "external_providers_dir") and config.external_providers_dir:
            registry = get_external_providers_from_dir(registry, config)
        # else lets check for modules in each provider
        registry = get_external_providers_from_module(
            registry=registry,
            config=config,
            building=(isinstance(config, BuildConfig) or isinstance(config, DistributionSpec)),
        )

    return registry


def get_external_providers_from_dir(
    registry: dict[Api, dict[str, ProviderSpec]], config
) -> dict[Api, dict[str, ProviderSpec]]:
    logger.warning(
        "Specifying external providers via `external_providers_dir` is being deprecated. Please specify `module:` in the provider instead."
    )
    external_providers_dir = os.path.abspath(os.path.expanduser(config.external_providers_dir))
    if not os.path.exists(external_providers_dir):
        raise FileNotFoundError(f"External providers directory not found: {external_providers_dir}")
    logger.info(f"Loading external providers from {external_providers_dir}")

    for api in providable_apis():
        api_name = api.name.lower()

        # Process both remote and inline providers
        for provider_type in ["remote", "inline"]:
            api_dir = os.path.join(external_providers_dir, provider_type, api_name)
            if not os.path.exists(api_dir):
                logger.debug(f"No {provider_type} provider directory found for {api_name}")
                continue

            # Look for provider spec files in the API directory
            for spec_path in glob.glob(os.path.join(api_dir, "*.yaml")):
                provider_name = os.path.splitext(os.path.basename(spec_path))[0]
                logger.info(f"Loading {provider_type} provider spec from {spec_path}")

                try:
                    with open(spec_path) as f:
                        spec_data = yaml.safe_load(f)

                    if provider_type == "remote":
                        spec = _load_remote_provider_spec(spec_data, api)
                        provider_type_key = f"remote::{provider_name}"
                    else:
                        spec = _load_inline_provider_spec(spec_data, api, provider_name)
                        provider_type_key = f"inline::{provider_name}"

                    logger.info(f"Loaded {provider_type} provider spec for {provider_type_key} from {spec_path}")
                    if provider_type_key in registry[api]:
                        logger.warning(f"Overriding already registered provider {provider_type_key} for {api.name}")
                    registry[api][provider_type_key] = spec
                    logger.info(f"Successfully loaded external provider {provider_type_key}")
                except yaml.YAMLError as yaml_err:
                    logger.error(f"Failed to parse YAML file {spec_path}: {yaml_err}")
                    raise yaml_err
                except Exception as e:
                    logger.error(f"Failed to load provider spec from {spec_path}: {e}")
                    raise e

    return registry


def get_external_providers_from_module(
    registry: dict[Api, dict[str, ProviderSpec]], config, building: bool
) -> dict[Api, dict[str, ProviderSpec]]:
    provider_list = None
    if isinstance(config, BuildConfig):
        provider_list = config.distribution_spec.providers.items()
    else:
        provider_list = config.providers.items()
    if provider_list is None:
        logger.warning("Could not get list of providers from config")
        return registry
    for provider_api, providers in provider_list:
        for provider in providers:
            if not hasattr(provider, "module") or provider.module is None:
                continue
            # get provider using module
            try:
                if not building:
                    package_name = provider.module.split("==")[0]
                    module = importlib.import_module(f"{package_name}.provider")
                    # if config class is wrong you will get an error saying module could not be imported
                    spec = module.get_provider_spec()
                else:
                    # pass in a partially filled out provider spec to satisfy the registry -- knowing we will be overwriting it later upon build and run
                    # in the case we are building we CANNOT import this module of course because it has not been installed.
                    spec = ProviderSpec(
                        api=Api(provider_api),
                        provider_type=provider.provider_type,
                        is_external=True,
                        module=provider.module,
                        config_class="",
                    )
                provider_type = provider.provider_type
                if isinstance(spec, list):
                    # optionally allow people to pass inline and remote provider specs as a returned list.
                    # with the old method, users could pass in directories of specs using overlapping code
                    # we want to ensure we preserve that flexibility in this method.
                    logger.info(
                        f"Detected a list of external provider specs from {provider.module} adding all to the registry"
                    )
                    for provider_spec in spec:
                        if provider_spec.provider_type != provider.provider_type:
                            continue
                        logger.info(f"Adding {provider.provider_type} to registry")
                        registry[Api(provider_api)][provider.provider_type] = provider_spec
                else:
                    registry[Api(provider_api)][provider_type] = spec
            except ModuleNotFoundError as exc:
                raise ValueError(
                    "get_provider_spec not found. If specifying an external provider via `module` in the Provider spec, the Provider must have the `provider.get_provider_spec` module available"
                ) from exc
            except Exception as e:
                logger.error(f"Failed to load provider spec from module {provider.module}: {e}")
                raise e
    return registry
