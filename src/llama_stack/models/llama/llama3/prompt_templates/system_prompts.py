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
from datetime import datetime
from typing import Any

from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    ToolDefinition,
)

from .base import PromptTemplate, PromptTemplateGeneratorBase


class SystemDefaultGenerator(PromptTemplateGeneratorBase):
    def gen(self, *args, **kwargs) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            Cutting Knowledge Date: December 2023
            Today Date: {{ today }}
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {
                "today": datetime.now().strftime("%d %B %Y")  # noqa: DTZ005 - we don't care about timezones here since we are displaying the date
            },
        )

    def data_examples(self) -> list[Any]:
        return [None]


class BuiltinToolGenerator(PromptTemplateGeneratorBase):
    def _tool_breakdown(self, tools: list[ToolDefinition]):
        builtin_tools, custom_tools = [], []
        for dfn in tools:
            if isinstance(dfn.tool_name, BuiltinTool):
                builtin_tools.append(dfn)
            else:
                custom_tools.append(dfn)

        return builtin_tools, custom_tools

    def gen(self, tools: list[ToolDefinition]) -> PromptTemplate:
        builtin_tools, custom_tools = self._tool_breakdown(tools)
        template_str = textwrap.dedent(
            """
            {% if builtin_tools or custom_tools -%}
            Environment: ipython
            {% endif -%}
            {% set builtin_tools = builtin_tools | reject('equalto', 'code_interpreter') | list -%}
            {% if builtin_tools -%}
            Tools: {{ builtin_tools | join(", ") | trim -}}
            {% endif %}
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {
                "builtin_tools": [t.tool_name.value for t in builtin_tools],
                "custom_tools": custom_tools,
            },
        )

    def data_examples(self) -> list[list[ToolDefinition]]:
        return [
            # builtin tools
            [
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
                ToolDefinition(tool_name=BuiltinTool.brave_search),
                ToolDefinition(tool_name=BuiltinTool.wolfram_alpha),
            ],
            # only code interpretor
            [
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ],
        ]


class JsonCustomToolGenerator(PromptTemplateGeneratorBase):
    def gen(self, custom_tools: list[ToolDefinition]) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            Answer the user's question by making use of the following functions if needed.
            If none of the function can be used, please say so.
            Here is a list of functions in JSON format:
            {% for t in custom_tools -%}
            {# manually setting up JSON because jinja sorts keys in unexpected ways -#}
            {%- set tname = t.tool_name -%}
            {%- set tdesc = t.description -%}
            {%- set tprops = t.input_schema.get('properties', {}) -%}
            {%- set required_params = t.input_schema.get('required', []) -%}
            {
                "type": "function",
                "function": {
                    "name": "{{tname}}",
                    "description": "{{tdesc}}",
                    "parameters": {
                        "type": "object",
                        "properties": [
                            {%- for name, param in tprops.items() %}
                            {
                                "{{name}}": {
                                    "type": "object",
                                    "description": "{{param.get('description', '')}}"
                                }
                            }{% if not loop.last %},{% endif %}
                            {%- endfor %}
                        ],
                        "required": {{ required_params | tojson }}
                    }
                }
            }
            {% endfor %}
            Return function calls in JSON format.
            """
        )

        return PromptTemplate(
            template_str.lstrip("\n"),
            {"custom_tools": [t.model_dump() for t in custom_tools]},
        )

    def data_examples(self) -> list[list[ToolDefinition]]:
        return [
            [
                ToolDefinition(
                    tool_name="trending_songs",
                    description="Returns the trending songs on a Music site",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "int",
                                "description": "The number of songs to return",
                            },
                            "genre": {
                                "type": "str",
                                "description": "The genre of the songs to return",
                            },
                        },
                        "required": ["n"],
                    },
                ),
            ]
        ]


class FunctionTagCustomToolGenerator(PromptTemplateGeneratorBase):
    def gen(self, custom_tools: list[ToolDefinition]) -> PromptTemplate:
        template_str = textwrap.dedent(
            """
            You have access to the following functions:

            {% for t in custom_tools %}
            {#- manually setting up JSON because jinja sorts keys in unexpected ways -#}
            {%- set tname = t.tool_name -%}
            {%- set tdesc = t.description -%}
            {%- set tprops = t.input_schema.get('properties', {}) -%}
            {%- set modified_params = {} -%}
            {%- for key, value in tprops.items() -%}
                {%- set param_copy = value.copy() -%}
                {%- if 'default' in param_copy -%}
                    {%- set _ = param_copy.pop('default', None) -%}
                {%- endif -%}
                {%- set _ = modified_params.update({key: param_copy}) -%}
            {%- endfor -%}
            {%- set tparams = modified_params | tojson -%}
            Use the function '{{ tname }}' to '{{ tdesc }}':
            {"name": "{{tname}}", "description": "{{tdesc}}", "parameters": {{tparams}}}

            {% endfor -%}
            Think very carefully before calling functions.
            If you choose to call a function ONLY reply in the following format with no prefix or suffix:

            <function=example_function_name>{"example_name": "example_value"}</function>

            Reminder:
            - If looking for real time information use relevant functions before falling back to brave_search
            - Function calls MUST follow the specified format, start with <function= and end with </function>
            - Required parameters MUST be specified
            - Only call one function at a time
            - Put the entire function call reply on one line
            """
        )
        return PromptTemplate(
            template_str.lstrip("\n"),
            {"custom_tools": [t.model_dump() for t in custom_tools]},
        )

    def data_examples(self) -> list[list[ToolDefinition]]:
        return [
            [
                ToolDefinition(
                    tool_name="trending_songs",
                    description="Returns the trending songs on a Music site",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "int",
                                "description": "The number of songs to return",
                            },
                            "genre": {
                                "type": "str",
                                "description": "The genre of the songs to return",
                            },
                        },
                        "required": ["n"],
                    },
                ),
            ]
        ]


class PythonListCustomToolGenerator(PromptTemplateGeneratorBase):  # noqa: N801
    DEFAULT_PROMPT = textwrap.dedent(
        """
        You are a helpful assistant. You have access to functions, but you should only use them if they are required.
        You are an expert in composing functions. You are given a question and a set of possible functions.
        Based on the question, you may or may not need to make one function/tool call to achieve the purpose.

        If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
        If you decide to invoke a function, you SHOULD NOT include any other text in the response. besides the function call in the above format.
        For a boolean parameter, be sure to use `True` or `False` (capitalized) for the value.


        {{ function_description }}
        """.strip("\n")
    )

    def gen(self, custom_tools: list[ToolDefinition], system_prompt: str | None = None) -> PromptTemplate:
        system_prompt = system_prompt or self.DEFAULT_PROMPT
        return PromptTemplate(
            system_prompt,
            {"function_description": self._gen_function_description(custom_tools)},
        )

    def _gen_function_description(self, custom_tools: list[ToolDefinition]) -> str:
        template_str = textwrap.dedent(
            """
            Here is a list of functions in JSON format that you can invoke.

            [
                {% for t in tools -%}
                {# manually setting up JSON because jinja sorts keys in unexpected ways -#}
                {%- set tname = t.tool_name -%}
                {%- set tdesc = t.description -%}
                {%- set tprops = (t.input_schema or {}).get('properties', {}) -%}
                {%- set required_params = (t.input_schema or {}).get('required', []) -%}
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

            You can answer general questions or invoke tools when necessary.
            In addition to tool calls, you should also augment your responses by using the tool outputs.

            """
        )
        template = PromptTemplate(
            template_str.strip("\n"),
            {"tools": [t.model_dump() for t in custom_tools]},
        )
        rendered: str = template.render()
        return rendered

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
