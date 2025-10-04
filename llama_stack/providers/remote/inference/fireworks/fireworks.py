# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from fireworks.client import Fireworks

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    Inference,
    LogProbConfig,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
)
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    request_has_media,
)

from .config import FireworksImplConfig

logger = get_logger(name=__name__, category="inference::fireworks")


class FireworksInferenceAdapter(OpenAIMixin, Inference, NeedsRequestProviderData):
    embedding_model_metadata = {
        "nomic-ai/nomic-embed-text-v1.5": {"embedding_dimension": 768, "context_length": 8192},
        "accounts/fireworks/models/qwen3-embedding-8b": {"embedding_dimension": 4096, "context_length": 40960},
    }

    def __init__(self, config: FireworksImplConfig) -> None:
        ModelRegistryHelper.__init__(self)
        self.config = config
        self.allowed_models = config.allowed_models

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def get_api_key(self) -> str:
        config_api_key = self.config.api_key.get_secret_value() if self.config.api_key else None
        if config_api_key:
            return config_api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.fireworks_api_key:
                raise ValueError(
                    'Pass Fireworks API Key in the header X-LlamaStack-Provider-Data as { "fireworks_api_key": <your api key>}'
                )
            return provider_data.fireworks_api_key

    def get_base_url(self) -> str:
        return "https://api.fireworks.ai/inference/v1"

    def _get_client(self) -> Fireworks:
        fireworks_api_key = self.get_api_key()
        return Fireworks(api_key=fireworks_api_key)

    def _build_options(
        self,
        sampling_params: SamplingParams | None,
        fmt: ResponseFormat | None,
        logprobs: LogProbConfig | None,
    ) -> dict:
        options = get_sampling_options(sampling_params)
        options.setdefault("max_tokens", 512)

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                options["response_format"] = {
                    "type": "grammar",
                    "grammar": fmt.bnf,
                }
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if logprobs and logprobs.top_k:
            options["logprobs"] = logprobs.top_k
            if options["logprobs"] <= 0 or options["logprobs"] >= 5:
                raise ValueError("Required range: 0 < top_k < 5")

        return options

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}
        media_present = request_has_media(request)

        llama_model = self.get_llama_model(request.model)
        # TODO: tools are never added to the request, so we need to add them here
        if media_present or not llama_model:
            input_dict["messages"] = [await convert_message_to_openai_dict(m, download=True) for m in request.messages]
        else:
            input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)

        # Fireworks always prepends with BOS
        if "prompt" in input_dict:
            if input_dict["prompt"].startswith("<|begin_of_text|>"):
                input_dict["prompt"] = input_dict["prompt"][len("<|begin_of_text|>") :]

        params = {
            "model": request.model,
            **input_dict,
            "stream": bool(request.stream),
            **self._build_options(request.sampling_params, request.response_format, request.logprobs),
        }
        logger.debug(f"params to fireworks: {params}")

        return params
