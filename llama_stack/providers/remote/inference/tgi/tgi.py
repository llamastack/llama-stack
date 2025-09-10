# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from collections.abc import AsyncGenerator

from huggingface_hub import AsyncInferenceClient, HfApi

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    OpenAIEmbeddingsResponse,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    OpenAICompletionToLlamaStackMixin,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_model_input_info,
    completion_request_to_prompt_model_input_info,
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
    Inference,
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
    ModelsProtocolPrivate,
):
    client: AsyncInferenceClient
    max_tokens: int
    model_id: str

    def __init__(self) -> None:
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.huggingface_repo_to_llama_model_id = {
            model.huggingface_repo: model.descriptor() for model in all_registered_models() if model.huggingface_repo
        }

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        model = await self.register_helper.register_model(model)
        if model.provider_resource_id != self.model_id:
            raise ValueError(
                f"Model {model.provider_resource_id} does not match the model {self.model_id} served by TGI."
            )
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

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

    async def _get_params_for_completion(self, request: CompletionRequest) -> dict:
        prompt, input_tokens = await completion_request_to_prompt_model_input_info(request)

        return dict(
            prompt=prompt,
            stream=request.stream,
            details=True,
            max_new_tokens=self._get_max_new_tokens(request.sampling_params, input_tokens),
            stop_sequences=["<|eom_id|>", "<|eot_id|>"],
            **self._build_options(request.sampling_params, request.response_format),
        )

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params_for_completion(request)

        async def _generate_and_convert_to_openai_compat():
            s = await self.client.text_generation(**params)
            async for chunk in s:
                token_result = chunk.token
                finish_reason = None
                if chunk.details:
                    finish_reason = chunk.details.finish_reason

                choice = OpenAICompatCompletionChoice(text=token_result.text, finish_reason=finish_reason)
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params_for_completion(request)
        r = await self.client.text_generation(**params)

        choice = OpenAICompatCompletionChoice(
            finish_reason=r.details.finish_reason,
            text="".join(t.text for t in r.details.tokens),
        )

        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )

        return process_completion_response(response)

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
        r = await self.client.text_generation(**params)

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
            s = await self.client.text_generation(**params)
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

    async def embeddings(
        self,
        model_id: str,
        contents: list[str] | list[InterleavedContentItem],
        text_truncation: TextTruncation | None = TextTruncation.none,
        output_dimension: int | None = None,
        task_type: EmbeddingTaskType | None = None,
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

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
        self.client = AsyncInferenceClient(model=config.url, provider="hf-inference")
        endpoint_info = await self.client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]


class InferenceAPIAdapter(_HfAdapter):
    async def initialize(self, config: InferenceAPIImplConfig) -> None:
        self.client = AsyncInferenceClient(model=config.huggingface_repo, token=config.api_token.get_secret_value())
        endpoint_info = await self.client.get_endpoint_info()
        self.max_tokens = endpoint_info["max_total_tokens"]
        self.model_id = endpoint_info["model_id"]


class InferenceEndpointAdapter(_HfAdapter):
    async def initialize(self, config: InferenceEndpointImplConfig) -> None:
        # Get the inference endpoint details
        api = HfApi(token=config.api_token.get_secret_value())
        endpoint = api.get_inference_endpoint(config.endpoint_name)
        # Wait for the endpoint to be ready (if not already)
        endpoint.wait(timeout=60)

        # Initialize the adapter
        self.client = endpoint.async_client
        self.model_id = endpoint.repository
        self.max_tokens = int(endpoint.raw["model"]["image"]["custom"]["env"]["MAX_TOTAL_TOKENS"])
