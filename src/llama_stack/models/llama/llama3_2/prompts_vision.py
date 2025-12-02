# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import textwrap
from pathlib import Path

from llama_stack.models.llama.datatypes import (
    RawMediaItem,
    RawMessage,
    RawTextItem,
)

from ..prompt_format import (
    TextCompletionContent,
    UseCase,
    llama3_1_builtin_tool_call_dialog,
    # llama3_1_builtin_tool_call_with_image_dialog,
    llama3_2_user_assistant_conversation,
)


def usecases():
    this_dir = Path(__file__).parent.parent.resolve()
    with open(this_dir / "scripts/resources/dog.jpg", "rb") as f:
        img = f.read()

    return [
        llama3_2_user_assistant_conversation(),
        UseCase(
            title="User and assistant conversation with Images",
            description="This example shows how to pass and image to the model as part of the messages.",
            dialogs=[
                [
                    RawMessage(
                        role="user",
                        content=[
                            RawMediaItem(data=img),
                            RawTextItem(text="Describe this image in two sentences"),
                        ],
                    )
                ],
            ],
            notes=textwrap.dedent(
                """
                - The `<|image|>` tag is used to indicate presence of the image
                - The model isn't an early fusion model so doesn't actually translate an image into several tokens. Instead the cross-attention layers take input "on the side" from a vision encoder
                ![Image](mm-model.png)
                - Its important to postion the <|image|> tag appropriately in the prompt. Image will only attend to the subsequent text tokens
                - The <|image|> tag is part of the user message body, implying that it should only come after the header `<|start_header_id|>{role}<|end_header_id|>` in the message body
                - We recommend using a single image in one prompt
                """
            ),
        ),
        UseCase(
            title="Builtin and Zero Shot Tool Calling",
            description=textwrap.dedent(
                """
                Llama3.2 vision models follow the same tool calling format as Llama3.1 models when inputs are text only.
                Use `Environment: ipython` to enable tools.
                Add `Tools: {{tool_name1}},{{tool_name2}}` for each of the builtin tools.
                The same builtin tools as Llama3.1 are available,
                - code_interpreter (for executing python code)
                - brave_search (to search the web)
                - wolfram_alpha (for querying wolfram alpha for mathematical questions)
                """,
            ),
            dialogs=[llama3_1_builtin_tool_call_dialog()],
            notes=textwrap.dedent(
                """
                - Note the `<|python_tag|>` before `brave_search` function call.
                - The `<|eom_id|>` tag is used to indicate the end of the message.
                - Similar to Llama3.1, code_interpreter is not explicitly mentioned but is enabled via `Environment: ipython`.
                - Tool Calling does NOT work with images in the prompt as of now.
                """
            ),
        ),
        # UseCase(
        #     title="Tool Calling for vision models",
        #     description=textwrap.dedent(
        #         """
        #         While Llama3.2 vision models follow the same tool calling format as Llama3.1 models when inputs are text only,
        #         they are not able to do tool calling when prompt contains image inputs (along with text).
        #         The recommended way would be to separate out the image understanding from the tool calling in successive prompts.
        #         Here is an example of how that could be done,
        #         """,
        #     ),
        #     dialogs=[llama3_1_builtin_tool_call_with_image_dialog()],
        #     notes=textwrap.dedent(
        #         """
        #         - Instead of a single prompt (image understanding + tool call), we split into two prompts to achieve the same result.
        #         """
        #     ),
        # ),
        UseCase(
            title="Prompt format for base models",
            description=textwrap.dedent(
                """
                For base models (Llama3.2-11B-Vision and Llama3.2-90B-Vision), the prompt format for a simple completion is as follows
                """
            ),
            dialogs=[
                TextCompletionContent(content="The color of the sky is blue but sometimes it can also be"),
            ],
            notes="- Same as Llama3.1",
        ),
        UseCase(
            title="Prompt format for base models with Image",
            description=textwrap.dedent(
                """
                For base models (Llama3.2-11B-Vision and Llama3.2-90B-Vision), here is an example of how the text completion format looks with an image,
                """
            ),
            dialogs=[
                TextCompletionContent(
                    content=[
                        RawMediaItem(data=img),
                        RawTextItem(text="If I had to write a haiku for this one"),
                    ]
                ),
            ],
            notes="- Note the placement of the special tags <|begin_of_text|> and <|image|>",
        ),
    ]
