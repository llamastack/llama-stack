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

from llama_stack.models.llama.datatypes import ToolDefinition
from llama_stack.models.llama.llama3.prompt_templates.base import (
    PromptTemplate,
    PromptTemplateGeneratorBase,
)


class PythonListCustomToolGenerator(PromptTemplateGeneratorBase):  # noqa: N801
    DEFAULT_PROMPT = textwrap.dedent(
        """
        You are a helpful assistant and an expert in function composition. You can answer general questions using your internal knowledge OR invoke functions when necessary. Follow these strict guidelines:

        1. FUNCTION CALLS:
        - ONLY use functions that are EXPLICITLY listed in the function list below
        - If NO functions are listed (empty function list []), respond ONLY with internal knowledge or "I don't have access to [Unavailable service] information"
        - If a function is not in the list, respond ONLY with internal knowledge or "I don't have access to [Unavailable service] information"
        - If ALL required parameters are present AND the query EXACTLY matches a listed function's purpose: output ONLY the function call(s)
        - Use exact format: [func_name1(param1=value1, param2=value2), func_name2(...)]
        Examples:
        CORRECT: [get_weather(location="Vancouver"), calculate_route(start="Boston", end="New York")] <- Only if get_weather and calculate_route are in function list
        INCORRECT: get_weather(location="New York")
        INCORRECT: Let me check the weather: [get_weather(location="New York")]
        INCORRECT: [get_events(location="Singapore")] <- If function not in list

        2. RESPONSE RULES:
        - For pure function requests matching a listed function: ONLY output the function call(s)
        - For knowledge questions: ONLY output text
        - For missing parameters: ONLY request the specific missing parameters
        - For unavailable services (not in function list): output ONLY with internal knowledge or "I don't have access to [Unavailable service] information". Do NOT execute a function call.
        - If the query asks for information beyond what a listed function provides: output ONLY with internal knowledge about your limitations
        - NEVER combine text and function calls in the same response
        - NEVER suggest alternative functions when the requested service is unavailable
        - NEVER create or invent new functions not listed below

        3. STRICT BOUNDARIES:
        - ONLY use functions from the list below - no exceptions
        - NEVER use a function as an alternative to unavailable information
        - NEVER call functions not present in the function list
        - NEVER add explanatory text to function calls
        - NEVER respond with empty brackets
        - Use proper Python/JSON syntax for function calls
        - Check the function list carefully before responding

        4. TOOL RESPONSE HANDLING:
        - When receiving tool responses: provide concise, natural language responses
        - Don't repeat tool response verbatim
        - Don't add supplementary information

        {{ function_description }}
        """.strip("\n")
    )

    def gen(self, custom_tools: list[ToolDefinition], system_prompt: str | None = None) -> PromptTemplate:
        system_prompt = system_prompt or self.DEFAULT_PROMPT
        return PromptTemplate(
            system_prompt,
            {"function_description": self._gen_function_description(custom_tools)},
        )

    def _gen_function_description(self, custom_tools: list[ToolDefinition]) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            Here is a list of functions in JSON format that you can invoke:
            [
                {% for t in tools -%}
                {# manually setting up JSON because jinja sorts keys in unexpected ways -#}
                {%- set tname = t.tool_name -%}
                {%- set tdesc = t.description -%}
                {%- set tprops = t.input_schema.get('properties', {}) -%}
                {%- set required_params = t.input_schema.get('required', []) -%}
                {
                    "name": "{{tname}}",
                    "description": "{{tdesc}}",
                    "parameters": {
                        "type": "dict",
                        "required": {{ required_params | tojson }},
                        "properties": {
                            {%- for name, param in tprops.items() %}
                            "{{name}}": {
                                "type": "{{param.get('type', 'string')}}",
                                "description": "{{param.get('description', '')}}"{% if param.get('default') %},
                                "default": "{{param.get('default')}}"{% endif %}
                            }{% if not loop.last %},{% endif %}
                            {%- endfor %}
                        }
                    }
                }{% if not loop.last %},
                {% endif -%}
                {%- endfor %}
            ]
            """
        )
        return PromptTemplate(
            template_str.strip("\n"),
            {"tools": [t.model_dump() for t in custom_tools]},
        ).render()

    def data_examples(self) -> list[list[ToolDefinition]]:
        return [
            [
                ToolDefinition(
                    tool_name="get_weather",
                    description="Get weather info for places",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city to get the weather for",
                            },
                            "metric": {
                                "type": "string",
                                "description": "The metric for weather. Options are: celsius, fahrenheit",
                                "default": "celsius",
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ]
        ]
