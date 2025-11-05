# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.distributions.template import BuildProvider, DistributionTemplate

from ..starter.starter import get_distribution_template as get_starter_distribution_template


def get_distribution_template() -> DistributionTemplate:
    template = get_starter_distribution_template(name="starter-gpu")
    template.description = "Quick start template for running Llama Stack with several popular providers. This distribution is intended for GPU-enabled environments."

    template.providers["post_training"] = [
        BuildProvider(provider_type="inline::huggingface-gpu"),
    ]
    return template
