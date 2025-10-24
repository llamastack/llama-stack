# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import textwrap

from .base import PromptTemplate, PromptTemplateGeneratorBase


class ToolResponseGenerator(PromptTemplateGeneratorBase):
    def gen(
        self,
        status: str,
        stdout: str | None = None,
        stderr: str | None = None,
    ):
        assert status in [
            "success",
            "failure",
        ], f"status must be 'success' or 'failure'; Got: {status}"
        template_str = textwrap.dedent(
            """
            {% if status == "success" %}completed{% else %}failed{% endif %}
            {%- if stdout %}
            [stdout]{{ stdout }}[/stdout]
            {%- endif -%}
            {%- if stderr %}
            [stderr]{{ stderr }}[/stderr]
            {%- endif -%}
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {
                "status": status,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def data_examples(self):
        return [
            # success
            {
                "status": "success",
                "stdout": '{"results":["something something"]}',
            },
            # failure
            {
                "status": "failure",
                "stderr": "brave_search encounter an error: could not communicate with api.brave.com",
            },
        ]
