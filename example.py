# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

os.environ["NVIDIA_API_KEY"] = "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"
# Option 1: Use default NIM URL (will auto-switch to ai.api.nvidia.com for rerank)
# os.environ["NVIDIA_BASE_URL"] = "https://ai.api.nvidia.com"
# Option 2: Use AI Foundation URL directly for rerank models
# os.environ["NVIDIA_BASE_URL"] = "https://ai.api.nvidia.com/v1"
os.environ["NVIDIA_BASE_URL"] = "https://integrate.api.nvidia.com"

import base64
import io
from PIL import Image

from llama_stack.core.library_client import LlamaStackAsLibraryClient

client = LlamaStackAsLibraryClient("nvidia")
client.initialize()

# # response = client.inference.completion(
# #     model_id="meta/llama-3.1-8b-instruct",
# #     content="Complete the sentence using one word: Roses are red, violets are :",
# #     stream=False,
# #     sampling_params={
# #         "max_tokens": 50,
# #     },
# # )
# # print(f"Response: {response.content}")


# response = client.inference.chat_completion(
#     model_id="nvidia/nvidia-nemotron-nano-9b-v2",
#     messages=[
#         {
#             "role": "system",
#             "content": "/think",
#         },
#         {
#             "role": "user",
#             "content": "How are you?",
#         },
#     ],
#     stream=False,
#     sampling_params={
#         "max_tokens": 1024,
#     },
# )
# print(f"Response: {response}")


print(client.models.list())
rerank_response = client.inference.rerank(
    model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
    query="query",
    items=[
        "item_1",
        "item_2",
        "item_3",
    ]
)

print(rerank_response)
for i, result in enumerate(rerank_response):
    print(f"{i+1}. [Index: {result.index}, "
          f"Score: {(result.relevance_score):.3f}]")

# # from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition

# # tool_definition = ToolDefinition(
# #     tool_name="get_weather",
# #     description="Get current weather information for a location",
# #     parameters={
# #         "location": ToolParamDefinition(
# #             param_type="string",
# #             description="The city and state, e.g. San Francisco, CA",
# #             required=True
# #         ),
# #         "unit": ToolParamDefinition(
# #             param_type="string",
# #             description="Temperature unit (celsius or fahrenheit)",
# #             required=False,
# #             default="celsius"
# #         )
# #     }
# # )

# # # tool_response = client.inference.chat_completion(
# # #     model_id="meta-llama/Llama-3.1-8B-Instruct",
# # #     messages=[
# # #         {"role": "user", "content": "What's the weather like in San Francisco?"}
# # #     ],
# # #     tools=[tool_definition],
# # # )

# # # print(f"Tool Response: {tool_response.completion_message.content}")
# # # if tool_response.completion_message.tool_calls:
# # #     for tool_call in tool_response.completion_message.tool_calls:
# # #         print(f"Tool Called: {tool_call.tool_name}")
# # #         print(f"Arguments: {tool_call.arguments}")


# # # from llama_stack.apis.inference import JsonSchemaResponseFormat, ResponseFormatType

# # # person_schema = {
# # #     "type": "object",
# # #     "properties": {
# # #         "name": {"type": "string"},
# # #         "age": {"type": "integer"},
# # #         "occupation": {"type": "string"},
# # #     },
# # #     "required": ["name", "age", "occupation"]
# # # }

# # # response_format = JsonSchemaResponseFormat(
# # #     type=ResponseFormatType.json_schema,
# # #     json_schema=person_schema
# # # )

# # # structured_response = client.inference.chat_completion(
# # #     model_id="meta-llama/Llama-3.1-8B-Instruct",
# # #     messages=[
# # #         {
# # #             "role": "user",
# # #             "content": "Create a profile for a fictional person named Alice who is 30 years old and is a software engineer. "
# # #         }
# # #     ],
# # #     response_format=response_format,
# # # )

# # # print(f"Structured Response: {structured_response.completion_message.content}")

# # # print("\n" + "="*50)
# # # print("VISION LANGUAGE MODEL (VLM) EXAMPLE")
# # # print("="*50)

# # def load_image_as_base64(image_path):
# #     with open(image_path, "rb") as image_file:
# #         img_bytes = image_file.read()
# #         return base64.b64encode(img_bytes).decode("utf-8")

# # image_path = "/home/jiayin/llama-stack/docs/dog.jpg"
# # demo_image_b64 = load_image_as_base64(image_path)

# # vlm_response = client.inference.chat_completion(
# #     model_id="nvidia/vila",
# #     messages=[
# #         {
# #             "role": "user",
# #             "content": [
# #                 {
# #                     "type": "image",
# #                     "image": {
# #                         "data": demo_image_b64,
# #                     },
# #                 },
# #                 {
# #                     "type": "text",
# #                     "text": "Please describe what you see in this image in detail.",
# #                 },
# #             ],
# #         }
# #     ],
# # )

# # print(f"VLM Response: {vlm_response.completion_message.content}")

# # # print("\n" + "="*50)
# # # print("EMBEDDING EXAMPLE")
# # # print("="*50)

# # # # Embedding example
# # # embedding_response = client.inference.embeddings(
# # #     model_id="nvidia/llama-3.2-nv-embedqa-1b-v2",
# # #     contents=["Hello world", "How are you today?"],
# # #     task_type="query"
# # # )

# # # print(f"Number of embeddings: {len(embedding_response.embeddings)}")
# # # print(f"Embedding dimension: {len(embedding_response.embeddings[0])}")
# # # print(f"First few values: {embedding_response.embeddings[0][:5]}")

# # # # from openai import OpenAI

# # # # client = OpenAI(
# # # #   base_url = "http://10.176.230.61:8000/v1",
# # # #   api_key = "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"
# # # # )

# # # # # completion = client.completions.create(
# # # # #   model="meta/llama-3.1-405b-instruct",
# # # # #   prompt="How are you?",
# # # # #   temperature=0.2,
# # # # #   top_p=0.7,
# # # # #   max_tokens=1024,
# # # # #   stream=False
# # # # # )

# # # # # # completion = client.chat.completions.create(
# # # # # #   model="meta/llama-3.1-8b-instruct",
# # # # # #   messages=[{"role":"user","content":"hi"}],
# # # # # #   temperature=0.2,
# # # # # #   top_p=0.7,
# # # # # #   max_tokens=1024,
# # # # # #   stream=True
# # # # # # )

# # # # # for chunk in completion:
# # # # #   if chunk.choices[0].delta.content is not None:
# # # # #     print(chunk.choices[0].delta.content, end="")


# # # # # response = client.inference.completion(
# # # # #     model_id="meta/llama-3.1-8b-instruct",
# # # # #     content="Complete the sentence using one word: Roses are red, violets are :",
# # # # #     stream=False,
# # # # #     sampling_params={
# # # # #         "max_tokens": 50,
# # # # #     },
# # # # # )
# # # # # print(f"Response: {response.content}")




# from openai import OpenAI

# client = OpenAI(
#   base_url = "https://integrate.api.nvidia.com/v1",
#   api_key = "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"
# )

# completion = client.chat.completions.create(
#   model="nvidia/nvidia-nemotron-nano-9b-v2",
#   messages=[{"role":"system","content":"/think"}],
#   temperature=0.6,
#   top_p=0.95,
#   max_tokens=2048,
#   frequency_penalty=0,
#   presence_penalty=0,
#   stream=True,
#   extra_body={
#     "min_thinking_tokens": 1024,
#     "max_thinking_tokens": 2048
#   }
# )

# for chunk in completion:
#   reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
#   if reasoning:
#     print(reasoning, end="")
#   if chunk.choices[0].delta.content is not None:
#     print(chunk.choices[0].delta.content, end="")
