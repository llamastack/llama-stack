"""
Test-only inference provider that echoes PROVIDER_DATA_VAR in responses.

This provider is used to test context isolation between requests in end-to-end
scenarios with a real server.
"""

import json
from typing import AsyncIterator
from pydantic import BaseModel

from llama_stack.apis.inference import (
    Inference,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAICompletion,
    OpenAICompletionRequestWithExtraBody,
    OpenAIEmbeddingsRequestWithExtraBody,
    OpenAIEmbeddingsResponse,
)
from llama_stack.apis.models import Model
from llama_stack.core.request_headers import PROVIDER_DATA_VAR
from llama_stack_client.types.inference_chat_completion_chunk import (
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
)


class ContextEchoConfig(BaseModel):
    """Minimal config for the test provider."""
    pass


class ContextEchoInferenceProvider(Inference):
    """
    Test-only provider that echoes the current PROVIDER_DATA_VAR value.

    Used to detect context leaks between streaming requests in end-to-end tests.
    """

    def __init__(self, config: ContextEchoConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def list_models(self) -> list[Model]:
        return []

    async def openai_embeddings(
        self,
        params: OpenAIEmbeddingsRequestWithExtraBody,
    ) -> OpenAIEmbeddingsResponse:
        raise NotImplementedError("Embeddings not supported by test provider")

    async def openai_completion(
        self,
        params: OpenAICompletionRequestWithExtraBody,
    ) -> OpenAICompletion:
        raise NotImplementedError("Use openai_chat_completion instead")

    async def openai_chat_completion(
        self,
        params: OpenAIChatCompletionRequestWithExtraBody,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        """Echo the provider data context back in streaming chunks."""

        async def stream_with_context():
            # Read the current provider data from context
            # This is the KEY part - if context leaks, this will show old data
            provider_data = PROVIDER_DATA_VAR.get()

            # Create a JSON message with the provider data
            # The test will parse this to verify correct isolation
            message = json.dumps({
                "provider_data": provider_data,
                "test_marker": "context_echo"
            })

            # Yield a chunk with the provider data
            yield OpenAIChatCompletionChunk(
                id="context-echo-1",
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkChoiceDelta(
                            content=message,
                            role="assistant",
                        ),
                        index=0,
                        finish_reason=None,
                    )
                ],
                created=0,
                model=params.model,
                object="chat.completion.chunk",
            )

            # Final chunk with finish_reason
            yield OpenAIChatCompletionChunk(
                id="context-echo-2",
                choices=[
                    ChatCompletionChunkChoice(
                        delta=ChatCompletionChunkChoiceDelta(),
                        index=0,
                        finish_reason="stop",
                    )
                ],
                created=0,
                model=params.model,
                object="chat.completion.chunk",
            )

        if params.stream:
            return stream_with_context()
        else:
            # Non-streaming fallback
            provider_data = PROVIDER_DATA_VAR.get()
            message_content = json.dumps({
                "provider_data": provider_data,
                "test_marker": "context_echo"
            })

            from llama_stack_client.types.inference_chat_completion import (
                ChatCompletionChoice,
                ChatCompletionMessage,
            )

            return OpenAIChatCompletion(
                id="context-echo",
                choices=[
                    ChatCompletionChoice(
                        finish_reason="stop",
                        index=0,
                        message=ChatCompletionMessage(
                            content=message_content,
                            role="assistant",
                        ),
                    )
                ],
                created=0,
                model=params.model,
                object="chat.completion",
            )
