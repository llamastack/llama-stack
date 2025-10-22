# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import re
import uuid
from string import Template
from typing import Any

from llama_stack.apis.common.content_types import ImageContentItem, TextContentItem
from llama_stack.apis.inference import (
    Inference,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.safety.safety import ModerationObject, ModerationObjectResults
from llama_stack.apis.shields import Shield
from llama_stack.core.datatypes import Api
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import Role
from llama_stack.models.llama.sku_types import CoreModelId
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)

from .config import LlamaGuardConfig

CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"

SAFE_RESPONSE = "safe"

CAT_VIOLENT_CRIMES = "Violent Crimes"
CAT_NON_VIOLENT_CRIMES = "Non-Violent Crimes"
CAT_SEX_CRIMES = "Sex Crimes"
CAT_CHILD_EXPLOITATION = "Child Exploitation"
CAT_DEFAMATION = "Defamation"
CAT_SPECIALIZED_ADVICE = "Specialized Advice"
CAT_PRIVACY = "Privacy"
CAT_INTELLECTUAL_PROPERTY = "Intellectual Property"
CAT_INDISCRIMINATE_WEAPONS = "Indiscriminate Weapons"
CAT_HATE = "Hate"
CAT_SELF_HARM = "Self-Harm"
CAT_SEXUAL_CONTENT = "Sexual Content"
CAT_ELECTIONS = "Elections"
CAT_CODE_INTERPRETER_ABUSE = "Code Interpreter Abuse"


SAFETY_CATEGORIES_TO_CODE_MAP = {
    CAT_VIOLENT_CRIMES: "S1",
    CAT_NON_VIOLENT_CRIMES: "S2",
    CAT_SEX_CRIMES: "S3",
    CAT_CHILD_EXPLOITATION: "S4",
    CAT_DEFAMATION: "S5",
    CAT_SPECIALIZED_ADVICE: "S6",
    CAT_PRIVACY: "S7",
    CAT_INTELLECTUAL_PROPERTY: "S8",
    CAT_INDISCRIMINATE_WEAPONS: "S9",
    CAT_HATE: "S10",
    CAT_SELF_HARM: "S11",
    CAT_SEXUAL_CONTENT: "S12",
    CAT_ELECTIONS: "S13",
    CAT_CODE_INTERPRETER_ABUSE: "S14",
}
SAFETY_CODE_TO_CATEGORIES_MAP = {v: k for k, v in SAFETY_CATEGORIES_TO_CODE_MAP.items()}

DEFAULT_LG_V3_SAFETY_CATEGORIES = [
    CAT_VIOLENT_CRIMES,
    CAT_NON_VIOLENT_CRIMES,
    CAT_SEX_CRIMES,
    CAT_CHILD_EXPLOITATION,
    CAT_DEFAMATION,
    CAT_SPECIALIZED_ADVICE,
    CAT_PRIVACY,
    CAT_INTELLECTUAL_PROPERTY,
    CAT_INDISCRIMINATE_WEAPONS,
    CAT_HATE,
    CAT_SELF_HARM,
    CAT_SEXUAL_CONTENT,
    CAT_ELECTIONS,
]

# accept both CoreModelId and huggingface repo id
LLAMA_GUARD_MODEL_IDS = {
    CoreModelId.llama_guard_3_8b.value: "meta-llama/Llama-Guard-3-8B",
    "meta-llama/Llama-Guard-3-8B": "meta-llama/Llama-Guard-3-8B",
    CoreModelId.llama_guard_3_1b.value: "meta-llama/Llama-Guard-3-1B",
    "meta-llama/Llama-Guard-3-1B": "meta-llama/Llama-Guard-3-1B",
    CoreModelId.llama_guard_3_11b_vision.value: "meta-llama/Llama-Guard-3-11B-Vision",
    "meta-llama/Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision",
    CoreModelId.llama_guard_4_12b.value: "meta-llama/Llama-Guard-4-12B",
    "meta-llama/Llama-Guard-4-12B": "meta-llama/Llama-Guard-4-12B",
}

MODEL_TO_SAFETY_CATEGORIES_MAP = {
    "meta-llama/Llama-Guard-3-8B": DEFAULT_LG_V3_SAFETY_CATEGORIES + [CAT_CODE_INTERPRETER_ABUSE],
    "meta-llama/Llama-Guard-3-1B": DEFAULT_LG_V3_SAFETY_CATEGORIES,
    "meta-llama/Llama-Guard-3-11B-Vision": DEFAULT_LG_V3_SAFETY_CATEGORIES,
    # Llama Guard 4 uses the same categories as Llama Guard 3
    # source: https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard4/12B/MODEL_CARD.md
    "meta-llama/Llama-Guard-4-12B": DEFAULT_LG_V3_SAFETY_CATEGORIES,
}


PROMPT_TASK = "Task: Check if there is unsafe content in '$agent_type' messages in conversations according our safety policy with the below categories."

SAFETY_CATEGORIES = """

<BEGIN UNSAFE CONTENT CATEGORIES>
$categories
<END UNSAFE CONTENT CATEGORIES>
"""

PROMPT_CONVERSATION = """
<BEGIN CONVERSATION>

$conversations

<END CONVERSATION>
"""


PROMPT_INSTRUCTIONS = """
Provide your safety assessment for ONLY THE LAST $agent_type message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories."""


PROMPT_TEMPLATE = Template(f"{PROMPT_TASK}{SAFETY_CATEGORIES}{PROMPT_CONVERSATION}{PROMPT_INSTRUCTIONS}")

logger = get_logger(name=__name__, category="safety")


class LlamaGuardSafetyImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: LlamaGuardConfig, deps) -> None:
        self.config = config
        self.inference_api = deps[Api.inference]

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        model_id = shield.provider_resource_id
        if not model_id:
            raise ValueError("Llama Guard shield must have a model id")

    async def unregister_shield(self, identifier: str) -> None:
        # LlamaGuard doesn't need to do anything special for unregistration
        # The routing table handles the removal from the registry
        pass

    async def run_shield(
        self,
        shield_id: str,
        messages: list[OpenAIMessageParam],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Unknown shield {shield_id}")

        messages = messages.copy()
        # some shields like llama-guard require the first message to be a user message
        # since this might be a tool call, first role might not be user
        if len(messages) > 0 and messages[0].role != "user":
            messages[0] = OpenAIUserMessageParam(content=messages[0].content)

        # Use the inference API's model resolution instead of hardcoded mappings
        # This allows the shield to work with any registered model
        model_id = shield.provider_resource_id

        # Determine safety categories based on the model type
        # For known Llama Guard models, use specific categories
        if model_id in LLAMA_GUARD_MODEL_IDS:
            # Use the mapped model for categories but the original model_id for inference
            mapped_model = LLAMA_GUARD_MODEL_IDS[model_id]
            safety_categories = MODEL_TO_SAFETY_CATEGORIES_MAP.get(mapped_model, DEFAULT_LG_V3_SAFETY_CATEGORIES)
        else:
            # For unknown models, use default Llama Guard 3 8B categories
            safety_categories = DEFAULT_LG_V3_SAFETY_CATEGORIES + [CAT_CODE_INTERPRETER_ABUSE]

        impl = LlamaGuardShield(
            model=model_id,
            inference_api=self.inference_api,
            excluded_categories=self.config.excluded_categories,
            safety_categories=safety_categories,
        )

        return await impl.run(messages)

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        if model is None:
            raise ValueError("Llama Guard moderation requires a model identifier.")

        if isinstance(input, list):
            messages = input.copy()
        else:
            messages = [input]

        # convert to user messages format with role
        messages = [OpenAIUserMessageParam(content=m) for m in messages]

        # Determine safety categories based on the model type
        # For known Llama Guard models, use specific categories
        if model in LLAMA_GUARD_MODEL_IDS:
            # Use the mapped model for categories but the original model_id for inference
            mapped_model = LLAMA_GUARD_MODEL_IDS[model]
            safety_categories = MODEL_TO_SAFETY_CATEGORIES_MAP.get(mapped_model, DEFAULT_LG_V3_SAFETY_CATEGORIES)
        else:
            # For unknown models, use default Llama Guard 3 8B categories
            safety_categories = DEFAULT_LG_V3_SAFETY_CATEGORIES + [CAT_CODE_INTERPRETER_ABUSE]

        impl = LlamaGuardShield(
            model=model,
            inference_api=self.inference_api,
            excluded_categories=self.config.excluded_categories,
            safety_categories=safety_categories,
        )

        return await impl.run_moderation(messages)


class LlamaGuardShield:
    def __init__(
        self,
        model: str,
        inference_api: Inference,
        excluded_categories: list[str] | None = None,
        safety_categories: list[str] | None = None,
    ):
        if excluded_categories is None:
            excluded_categories = []
        if safety_categories is None:
            safety_categories = []

        assert len(excluded_categories) == 0 or all(
            x in SAFETY_CATEGORIES_TO_CODE_MAP.values() for x in excluded_categories
        ), "Invalid categories in excluded categories. Expected format is ['S1', 'S2', ..]"

        self.model = model
        self.inference_api = inference_api
        self.excluded_categories = excluded_categories
        self.safety_categories = safety_categories

    def check_unsafe_response(self, response: str) -> str | None:
        match = re.match(r"^unsafe\n(.*)$", response)
        if match:
            # extracts the unsafe code
            extracted = match.group(1)
            return extracted

        return None

    def get_safety_categories(self) -> list[str]:
        excluded_categories = self.excluded_categories
        if set(excluded_categories) == set(SAFETY_CATEGORIES_TO_CODE_MAP.values()):
            excluded_categories = []

        final_categories = []

        all_categories = self.safety_categories
        for cat in all_categories:
            cat_code = SAFETY_CATEGORIES_TO_CODE_MAP[cat]
            if cat_code in excluded_categories:
                continue
            final_categories.append(f"{cat_code}: {cat}.")

        return final_categories

    def validate_messages(self, messages: list[OpenAIMessageParam]) -> list[OpenAIMessageParam]:
        if len(messages) == 0:
            raise ValueError("Messages must not be empty")
        if messages[0].role != Role.user.value:
            raise ValueError("Messages must start with user")

        if len(messages) >= 2 and (messages[0].role == Role.user.value and messages[1].role == Role.user.value):
            messages = messages[1:]

        return messages

    async def run(self, messages: list[OpenAIMessageParam]) -> RunShieldResponse:
        messages = self.validate_messages(messages)

        if self.model == CoreModelId.llama_guard_3_11b_vision.value:
            shield_input_message = self.build_vision_shield_input(messages)
        else:
            shield_input_message = self.build_text_shield_input(messages)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=self.model,
            messages=[shield_input_message],
            stream=False,
            temperature=0.0,  # default is 1, which is too high for safety
        )
        response = await self.inference_api.openai_chat_completion(params)
        content = response.choices[0].message.content
        content = content.strip()
        return self.get_shield_response(content)

    def build_text_shield_input(self, messages: list[OpenAIMessageParam]) -> OpenAIUserMessageParam:
        return OpenAIUserMessageParam(content=self.build_prompt(messages))

    def build_vision_shield_input(self, messages: list[OpenAIMessageParam]) -> OpenAIUserMessageParam:
        conversation = []
        most_recent_img = None

        for m in messages[::-1]:
            if isinstance(m.content, str) or isinstance(m.content, TextContentItem):
                conversation.append(m)
            elif isinstance(m.content, ImageContentItem):
                if most_recent_img is None and m.role == Role.user.value:
                    most_recent_img = m.content
                    conversation.append(m)
            elif isinstance(m.content, list):
                content = []
                for c in m.content:
                    if isinstance(c, str) or isinstance(c, TextContentItem):
                        content.append(c)
                    elif isinstance(c, ImageContentItem):
                        if most_recent_img is None and m.role == Role.user.value:
                            most_recent_img = c
                            content.append(c)
                    else:
                        raise ValueError(f"Unknown content type: {c}")

                conversation.append(OpenAIUserMessageParam(content=content))
            else:
                raise ValueError(f"Unknown content type: {m.content}")

        prompt = []
        if most_recent_img is not None:
            prompt.append(most_recent_img)
        prompt.append(self.build_prompt(conversation[::-1]))

        return OpenAIUserMessageParam(content=prompt)

    def build_prompt(self, messages: list[OpenAIMessageParam]) -> str:
        categories = self.get_safety_categories()
        categories_str = "\n".join(categories)
        conversations_str = "\n\n".join(
            [f"{m.role.capitalize()}: {interleaved_content_as_str(m.content)}" for m in messages]
        )
        return PROMPT_TEMPLATE.substitute(
            agent_type=messages[-1].role.capitalize(),
            categories=categories_str,
            conversations=conversations_str,
        )

    def get_shield_response(self, response: str) -> RunShieldResponse:
        response = response.strip()
        if response == SAFE_RESPONSE:
            return RunShieldResponse(violation=None)

        unsafe_code = self.check_unsafe_response(response)
        if unsafe_code:
            unsafe_code_list = unsafe_code.split(",")
            if set(unsafe_code_list).issubset(set(self.excluded_categories)):
                return RunShieldResponse(violation=None)

            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=CANNED_RESPONSE_TEXT,
                    metadata={"violation_type": unsafe_code},
                ),
            )

        raise ValueError(f"Unexpected response: {response}")

    async def run_moderation(self, messages: list[OpenAIMessageParam]) -> ModerationObject:
        if not messages:
            return self.create_moderation_object(self.model)

        # TODO: Add Image based support for OpenAI Moderations
        shield_input_message = self.build_text_shield_input(messages)

        params = OpenAIChatCompletionRequestWithExtraBody(
            model=self.model,
            messages=[shield_input_message],
            stream=False,
            temperature=0.0,  # default is 1, which is too high for safety
        )
        response = await self.inference_api.openai_chat_completion(params)
        content = response.choices[0].message.content
        content = content.strip()
        return self.get_moderation_object(content)

    def create_moderation_object(self, model: str, unsafe_code: str | None = None) -> ModerationObject:
        """Create a ModerationObject for either safe or unsafe content.

        Args:
            model: The model name
            unsafe_code: Optional comma-separated list of safety codes. If None, creates safe object.

        Returns:
            ModerationObject with appropriate configuration
        """
        # Set default values for safe case
        categories = dict.fromkeys(SAFETY_CATEGORIES_TO_CODE_MAP.keys(), False)
        category_scores = dict.fromkeys(SAFETY_CATEGORIES_TO_CODE_MAP.keys(), 1.0)
        category_applied_input_types = {key: [] for key in SAFETY_CATEGORIES_TO_CODE_MAP.keys()}
        flagged = False
        user_message = None
        metadata = {}

        # Handle unsafe case
        if unsafe_code:
            unsafe_code_list = [code.strip() for code in unsafe_code.split(",")]
            invalid_codes = [code for code in unsafe_code_list if code not in SAFETY_CODE_TO_CATEGORIES_MAP]
            if invalid_codes:
                logger.warning(f"Invalid safety codes returned: {invalid_codes}")
                # just returning safe object, as we don't know what the invalid codes can map to
                return ModerationObject(
                    id=f"modr-{uuid.uuid4()}",
                    model=model,
                    results=[
                        ModerationObjectResults(
                            flagged=flagged,
                            categories=categories,
                            category_applied_input_types=category_applied_input_types,
                            category_scores=category_scores,
                            user_message=user_message,
                            metadata=metadata,
                        )
                    ],
                )

            llama_guard_category = [SAFETY_CODE_TO_CATEGORIES_MAP[code] for code in unsafe_code_list]

            # Update categories for unsafe content
            categories = {k: k in llama_guard_category for k in SAFETY_CATEGORIES_TO_CODE_MAP.keys()}
            category_scores = {
                k: 1.0 if k in llama_guard_category else 0.0 for k in SAFETY_CATEGORIES_TO_CODE_MAP.keys()
            }
            category_applied_input_types = {
                k: ["text"] if k in llama_guard_category else [] for k in SAFETY_CATEGORIES_TO_CODE_MAP.keys()
            }
            flagged = True
            user_message = CANNED_RESPONSE_TEXT
            metadata = {"violation_type": unsafe_code_list}

        return ModerationObject(
            id=f"modr-{uuid.uuid4()}",
            model=model,
            results=[
                ModerationObjectResults(
                    flagged=flagged,
                    categories=categories,
                    category_applied_input_types=category_applied_input_types,
                    category_scores=category_scores,
                    user_message=user_message,
                    metadata=metadata,
                )
            ],
        )

    def is_content_safe(self, response: str, unsafe_code: str | None = None) -> bool:
        """Check if content is safe based on response and unsafe code."""
        if response.strip().lower().startswith(SAFE_RESPONSE):
            return True

        if unsafe_code:
            unsafe_code_list = unsafe_code.split(",")
            if set(unsafe_code_list).issubset(set(self.excluded_categories)):
                return True

        return False

    def get_moderation_object(self, response: str) -> ModerationObject:
        response = response.strip()
        if self.is_content_safe(response):
            return self.create_moderation_object(self.model)
        unsafe_code = self.check_unsafe_response(response)
        if not unsafe_code:
            raise ValueError(f"Unexpected response: {response}")

        if self.is_content_safe(response, unsafe_code):
            return self.create_moderation_object(self.model)
        else:
            return self.create_moderation_object(self.model, unsafe_code)
