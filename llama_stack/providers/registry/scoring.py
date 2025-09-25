# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.scoring,
            provider_type="inline::basic",
            pip_packages=["requests"],
            module="llama_stack.providers.inline.scoring.basic",
            config_class="llama_stack.providers.inline.scoring.basic.BasicScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            description="Basic scoring provider for simple evaluation metrics and scoring functions.",
        ),
        InlineProviderSpec(
            api=Api.scoring,
            provider_type="inline::llm-as-judge",
            pip_packages=[],
            module="llama_stack.providers.inline.scoring.llm_as_judge",
            config_class="llama_stack.providers.inline.scoring.llm_as_judge.LlmAsJudgeScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.inference,
            ],
            description="LLM-as-judge scoring provider that uses language models to evaluate and score responses.",
        ),
        InlineProviderSpec(
            api=Api.scoring,
            provider_type="inline::braintrust",
            pip_packages=["autoevals"],
            module="llama_stack.providers.inline.scoring.braintrust",
            config_class="llama_stack.providers.inline.scoring.braintrust.BraintrustScoringConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            provider_data_validator="llama_stack.providers.inline.scoring.braintrust.BraintrustProviderDataValidator",
            description="Braintrust scoring provider for evaluation and scoring using the Braintrust platform.",
        ),
    ]
