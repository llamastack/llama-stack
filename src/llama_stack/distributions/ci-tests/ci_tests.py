# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.distributions.template import DistributionTemplate
from llama_stack_api import ConnectorInput, ModelInput, ModelType

from ..starter.starter import get_distribution_template as get_starter_distribution_template


def get_distribution_template() -> DistributionTemplate:
    template = get_starter_distribution_template(name="ci-tests")
    template.description = "CI tests for Llama Stack"

    # Pre-register the Bedrock model.
    # Unlike other providers, Bedrock's OpenAI-compatible endpoint does not support
    # /v1/models for dynamic model discovery (see bedrock.py:list_provider_model_ids).
    # Models must be registered in config to be available.
    bedrock_model = ModelInput(
        model_id="openai.gpt-oss-20b-1:0",
        provider_id="bedrock",
        provider_model_id="openai.gpt-oss-20b-1:0",
        model_type=ModelType.llm,
        metadata={"description": "OpenAI GPT-OSS 20B on Bedrock (us-west-2)"},
    )

    # Pre-register a test MCP connector used by test_response_connector_resolution_mcp_tool.
    # The test starts an MCP server on port 5199 and references it by connector_id.
    test_mcp_connector = ConnectorInput(
        connector_id="test-mcp-connector",
        url="http://localhost:5199/sse",
    )

    # Add conditional authentication config (disabled by default for CI tests)
    # This tests the conditional auth provider feature and provides a template for users
    # To enable: export AUTH_PROVIDER=enabled and configure the auth env vars
    auth_config = {
        # Authentication is disabled by default (AUTH_PROVIDER not set)
        # To enable: export AUTH_PROVIDER=enabled
        # Then configure the required auth provider settings below
        "provider_config": {
            "type": "${env.AUTH_PROVIDER:+oauth2_token}",
            "audience": "${env.AUTH_AUDIENCE:=llama-stack}",
            "issuer": "${env.AUTH_ISSUER:=}",
            "jwks": {
                "uri": "${env.AUTH_JWKS_URI:=}",
                "key_recheck_period": "${env.AUTH_JWKS_RECHECK_PERIOD:=3600}",
            },
            "verify_tls": "${env.AUTH_VERIFY_TLS:=true}",
        }
    }

    for run_config in template.run_configs.values():
        if run_config.default_models is None:
            run_config.default_models = []
        run_config.default_models.append(bedrock_model)

        if run_config.default_connectors is None:
            run_config.default_connectors = []
        run_config.default_connectors.append(test_mcp_connector)

        # Add conditional auth config
        run_config.auth_config = auth_config

    return template
