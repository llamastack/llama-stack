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

from pydantic import BaseModel, Field

from llama_stack.models.llama.datatypes import (
    RawContent,
    RawMessage,
    ToolPromptFormat,
)
from llama_stack.models.llama.llama4.tokenizer import Tokenizer

from .llama3.interface import LLama31Interface
from .llama3.template_data import (
    system_message_builtin_code_only,
    system_message_builtin_tools_only,
    system_message_custom_tools_only,
)


class TextCompletionContent(BaseModel):
    content: RawContent = ""


class UseCase(BaseModel):
    title: str = ""
    description: str = ""
    dialogs: list[list[RawMessage] | TextCompletionContent | str] = Field(default_factory=list)
    notes: str = ""
    tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json
    max_gen_len: int = 512

    def md_format(self):
        section = textwrap.dedent(
            """
            ## {title}

            {description}

            {dialogs_text}
            {notes}

            """
        )
        return section.lstrip()

    def dialogs_to_text(self, generator) -> str:
        def _code_block(text):
            return f"```\n{text}\n```"

        text = ""
        for dialog in self.dialogs:
            if isinstance(dialog, str):
                text += dialog
                text += "\n\n"
                continue
            else:
                batch = [dialog]
                method = (
                    generator.completion if isinstance(dialog, TextCompletionContent) else generator.chat_completion
                )
                input_tokens = []
                output_tokens = []
                for token_results in method(batch, echo=True, temperature=0.1, top_p=0.95):
                    result = token_results[0]
                    if result.source == "input":
                        input_tokens.append(result.token)
                    else:
                        output_tokens.append(result.token)

                    if result.finished:
                        break
            text += "##### Input Prompt Format\n"

            # FIXME: This is added to undo the hack in chat_formatter where
            # vision tokens are replaced with 128256.
            input_tokens = [generator.formatter.vision_token if t == 128256 else t for t in input_tokens]

            text += _code_block(generator.tokenizer.decode(input_tokens))
            # TODO: Figure out if "↵" needs to be added for newlines or end or some indication
            text += "\n\n"
            text += "##### Model Response Format\n"
            text += _code_block(generator.tokenizer.decode(output_tokens))
            text += "\n\n"

        return text

    def to_text(self, generator):
        section = self.md_format()
        dialogs_text = self.dialogs_to_text(generator)
        notes = f"##### Notes\n{self.notes}" if self.notes else ""
        section = section.format(
            title=self.title,
            description=self.description,
            dialogs_text=dialogs_text,
            notes=notes,
        )
        return section


class Llama4UseCase(UseCase):
    def dialogs_to_text(self, generator) -> str:
        def _code_block(text):
            return f"```\n{text}\n```"

        text = ""
        tokenizer = Tokenizer.get_instance()
        for dialog in self.dialogs:
            if isinstance(dialog, str):
                text += dialog
                text += "\n\n"
                continue
            else:
                batch = [dialog]
                method = (
                    generator.completion if isinstance(dialog, TextCompletionContent) else generator.chat_completion
                )
                input_tokens = []
                output_tokens = []
                for token_results in method(batch, echo=True, temperature=0.0):
                    result = token_results[0]
                    if result.source == "input":
                        input_tokens.append(result.token)
                    else:
                        output_tokens.append(result.token)

                    if result.finished:
                        break

            text += "##### Input Prompt Format\n"
            text += _code_block(tokenizer.decode(input_tokens))
            text += "\n\n"
            text += "##### Model Response Format\n"
            text += _code_block(tokenizer.decode(output_tokens))
            text += "\n\n"

        return text


def llama3_1_builtin_tool_call_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_builtin_tools_only())
    messages += interface.user_message(content="Search the web for the latest price of 1oz gold?")

    return messages


def llama3_1_builtin_code_interpreter_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_builtin_code_only())
    messages += interface.user_message(
        content="Write code to check if number is prime. Use it to verify if number 7 is prime"
    )

    return messages


def llama3_1_custom_tool_call_dialog(tool_prompt_format=ToolPromptFormat.json):
    interface = LLama31Interface(tool_prompt_format)

    messages = interface.system_messages(**system_message_custom_tools_only())
    messages += interface.user_message(content="Use tools to get latest trending songs")
    return messages


def llama3_2_user_assistant_conversation():
    return UseCase(
        title="User and assistant conversation",
        description="Here is a regular multi-turn user assistant conversation and how its formatted.",
        dialogs=[
            [
                RawMessage(role="system", content="You are a helpful assistant"),
                RawMessage(role="user", content="Who are you?"),
            ]
        ],
        notes="This format is unchanged from Llama3.1",
    )
