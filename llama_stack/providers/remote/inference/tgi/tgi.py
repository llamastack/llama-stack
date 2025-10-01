# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncGenerator

from huggingface_hub import AsyncInferenceClient, HfApi
from pydantic import SecretStr

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Inference,
    LogProbConfig,
    Message,
    OpenAIEmbeddingsResponse,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model
from llama_stack.apis.models.models import ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.openai_mixin import OpenAIMixin
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_model_input_info,
)

from .config import InferenceAPIImplConfig, InferenceEndpointImplConfig, TGIImplConfig

log = get_logger(name=__name__, category="inference::tgi")


def build_hf_repo_model_entries():
    return [
        build_hf_repo_model_entry(
            model.huggingface_repo,
            model.descriptor(),
        )
        for model in all_registered_models()
        if model.huggingface_repo
    ]


class _HfAdapter(
    OpenAIMixin,
    Inference,
    ModelsProtocolPrivate,
):
    url: str
    api_key: SecretStr

    hf_client: AsyncInferenceClient
    max_tokens: int
    model_id: str

    overwrite_completion_id = True  # TGI always returns id=""

    def __init__(self) -> None:
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.huggingface_repo_to_llama_model_id = {
            model.huggingface_repo: model.descriptor() for model in all_registered_models() if model.huggingface_repo
        }

    def get_api_key(self):
        return self.api_key.get_secret_value()

    def get_base_url(self):
        return self.url

    async def shutdown(self) -> None:
        pass

    async def list_models(self) -> list[Model] | None:
        models = []
        async for model in self.client.models.list():
            models.append(
                Model(
                    identifier=model.id,
                    provider_resource_id=model.id,
                    provider_id=self.__provider_id__,
                    metadata={},
                    model_type=ModelType.llm,
                )
            )
        return models

    async def register_model(self, model: Model) -> Model:
        if model.provider_resource_id != self.model_id:
            raise ValueError(
                f"Model {model.provider_resource_id} does not match the model {self.model_id} served by TGI."
            )
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    def _get_max_new_tokens(self, sampling_params, input_tokens):
        return min(
            sampling_params.max_tokens or (self.max_tokens - input_tokens),
            self.max_tokens - input_tokens - 1,
        )

    def _build_options(
        self,
        sampling_params: SamplingParams | None = None,
        fmt: ResponseFormat = None,
    ):
        options = get_sampling_options(sampling_params)
        # TGI does not support temperature=0 when using greedy sampling
        # We set it to 1e-3 instead, anything lower outputs garbage from TGI
        # We can use top_p sampling strategy to specify lower temperature
        if abs(options["temperature"]) < 1e-10:
            options["temperature"] = 1e-3

        # delete key "max_tokens" from options since its not supported by the API
        options.pop("max_tokens", None)
        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["grammar"] = {
                    "type": "json",
                    "value": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise ValueError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unexpected response format: {fmt.type}")

        return options

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = await self.hf_client.text_generation(**params)

        choice = OpenAICompatCompletionChoice(
            finish_reason=r.details.finish_reason,
            text="".join(t.text for t in r.details.tokens),
        )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )
        return process_chat_completion_response(response, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            s = await self.hf_client.text_generation(**params)
            async for chunk in s:
                token_result = chunk.token

                choice = OpenAICompatCompletionChoice(text=token_result.text)
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        prompt, input_tokens = await chat_completion_request_to_model_input_info(
            request, self.register_helper.get_llama_model(request.model)
        )
        return dict(
            prompt=prompt,
            stream=request.stream,
            details=True,
            max_new_tokens=self._get_max_new_tokens(request.sampling_params, input_tokens),
            stop_sequences=["<|eom_id|>", "<|eot_id|>"],
            **self._build_options(request.sampling_params, request.response_format),
        )

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError()


class TGIAdapter(_HfAdapter):
    async def initialize(self, config: TGIImplConfig) -> None:
        if not config.url:
            raise ValueError("You must provide a URL in run.yaml (or via the TGI_URL environment variable) to use TGI.")
        log.info(f"Initializing TGI client with url={config.url}")
        self.hf_client = AsyncInferenceClient(model=config.url, provider="hf-inference")
        endpoint_info = await self.hf_client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]
        self.url = f"{config.url.rstrip('/')}/v1"
        self.api_key = SecretStr("NO_KEY")


class InferenceAPIAdapter(_HfAdapter):
    async def initialize(self, config: InferenceAPIImplConfig) -> None:
        self.hf_client = AsyncInferenceClient(model=config.huggingface_repo, token=config.api_token.get_secret_value())
        endpoint_info = await self.hf_client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]
        # TODO: how do we set url for this?


class InferenceEndpointAdapter(_HfAdapter):
    async def initialize(self, config: InferenceEndpointImplConfig) -> None:
        # Get the inference endpoint details
        api = HfApi(token=config.api_token.get_secret_value())
        endpoint = api.get_inference_endpoint(config.endpoint_name)
        # Wait for the endpoint to be ready (if not already)
        endpoint.wait(timeout=60)

        # Initialize the adapter
        self.hf_client = endpoint.async_client
        self.model_id = endpoint.repository
        self.max_tokens = int(endpoint.raw["model"]["image"]["custom"]["env"]["MAX_TOTAL_TOKENS"])
        # TODO: how do we set url for this?
