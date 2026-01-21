# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.distributions.template import DistributionTemplate
from llama_stack_api import ModelInput, ModelType

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

    # Add bedrock model to all run configs
    for run_config in template.run_configs.values():
        if run_config.default_models is None:
            run_config.default_models = []
        run_config.default_models.append(bedrock_model)

    return template
