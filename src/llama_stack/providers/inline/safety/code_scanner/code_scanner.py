# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeshield.cs import CodeShieldScanResult

from llama_stack.apis.inference import OpenAIMessageParam
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.safety.safety import ModerationObject, ModerationObjectResults
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)

from .config import CodeScannerConfig

log = get_logger(name=__name__, category="safety")

ALLOWED_CODE_SCANNER_MODEL_IDS = [
    "code-scanner",
    "code-shield",
]


class MetaReferenceCodeScannerSafetyImpl(Safety):
    def __init__(self, config: CodeScannerConfig, deps) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if shield.provider_resource_id not in ALLOWED_CODE_SCANNER_MODEL_IDS:
            raise ValueError(
                f"Unsupported Code Scanner ID: {shield.provider_resource_id}. Allowed IDs: {ALLOWED_CODE_SCANNER_MODEL_IDS}"
            )

    async def run_shield(
        self,
        shield_id: str,
        messages: list[OpenAIMessageParam],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        from codeshield.cs import CodeShield

        text = "\n".join([interleaved_content_as_str(m.content) for m in messages])
        log.info(f"Running CodeScannerShield on {text[50:]}")
        result = await CodeShield.scan_code(text)

        violation = None
        if result.is_insecure:
            violation = SafetyViolation(
                violation_level=(ViolationLevel.ERROR),
                user_message="Sorry, I found security concerns in the code.",
                metadata={"violation_type": ",".join([issue.pattern_id for issue in result.issues_found])},
            )
        return RunShieldResponse(violation=violation)

    def get_moderation_object_results(self, scan_result: "CodeShieldScanResult") -> ModerationObjectResults:
        categories = {}
        category_scores = {}
        category_applied_input_types = {}

        flagged = scan_result.is_insecure
        user_message = None
        metadata = {}

        if scan_result.is_insecure:
            pattern_ids = [issue.pattern_id for issue in scan_result.issues_found]
            categories = dict.fromkeys(pattern_ids, True)
            category_scores = dict.fromkeys(pattern_ids, 1.0)
            category_applied_input_types = {key: ["text"] for key in pattern_ids}
            user_message = f"Security concerns detected in the code. {scan_result.recommended_treatment.name}: {', '.join([issue.description for issue in scan_result.issues_found])}"
            metadata = {"violation_type": ",".join([issue.pattern_id for issue in scan_result.issues_found])}

        return ModerationObjectResults(
            flagged=flagged,
            categories=categories,
            category_scores=category_scores,
            category_applied_input_types=category_applied_input_types,
            user_message=user_message,
            metadata=metadata,
        )

    async def run_moderation(self, input: str | list[str], model: str | None = None) -> ModerationObject:
        if model is None:
            raise ValueError("Code scanner moderation requires a model identifier.")

        inputs = input if isinstance(input, list) else [input]
        results = []

        from codeshield.cs import CodeShield

        for text_input in inputs:
            log.info(f"Running CodeScannerShield moderation on input: {text_input[:100]}...")
            try:
                scan_result = await CodeShield.scan_code(text_input)
                moderation_result = self.get_moderation_object_results(scan_result)
            except Exception as e:
                log.error(f"CodeShield.scan_code failed: {e}")
                # create safe fallback response on scanner failure to avoid blocking legitimate requests
                moderation_result = ModerationObjectResults(
                    flagged=False,
                    categories={},
                    category_scores={},
                    category_applied_input_types={},
                    user_message=None,
                    metadata={"scanner_error": str(e)},
                )
            results.append(moderation_result)

        return ModerationObject(id=str(uuid.uuid4()), model=model, results=results)
