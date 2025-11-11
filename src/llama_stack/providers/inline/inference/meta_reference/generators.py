# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import math
from typing import Optional

import torch
from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData

from llama_stack.apis.inference import (
    GreedySamplingStrategy,
    JsonSchemaResponseFormat,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIResponseFormatJSONSchema,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    TopPSamplingStrategy,
)
from llama_stack.models.llama.datatypes import QuantizationMode, ToolPromptFormat
from llama_stack.models.llama.llama3.generation import Llama3
from llama_stack.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer
from llama_stack.models.llama.llama4.generation import Llama4
from llama_stack.models.llama.llama4.tokenizer import Tokenizer as Llama4Tokenizer
from llama_stack.models.llama.sku_types import Model, ModelFamily

from .common import model_checkpoint_dir
from .config import MetaReferenceInferenceConfig
from .inference import resolve_model

Tokenizer = Llama4Tokenizer | Llama3Tokenizer


class LogitsProcessor:
    def __init__(self, token_enforcer: TokenEnforcer):
        self.token_enforcer = token_enforcer
        self.mask: torch.Tensor | None = None

    def __call__(self, tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        token_sequence = tokens[0, :].tolist()
        allowed_tokens = self.token_enforcer.get_allowed_tokens(token_sequence)

        if self.mask is not None:
            self.mask.fill_(-math.inf)
        else:
            self.mask = torch.full_like(scores, -math.inf)

        self.mask[:, :, allowed_tokens] = 0
        scores = scores + self.mask
        return scores


def get_logits_processor(
    tokenizer: Tokenizer,
    vocab_size: int,
    response_format: ResponseFormat | None,
) -> Optional["LogitsProcessor"]:
    if response_format is None:
        return None

    if not isinstance(response_format, JsonSchemaResponseFormat):
        raise ValueError(f"Unsupported response format type {response_format.type}")

    parser = JsonSchemaParser(response_format.json_schema)
    data = TokenEnforcerTokenizerData(
        _build_regular_tokens_list(tokenizer, vocab_size),
        tokenizer.decode,
        tokenizer.stop_tokens,
    )
    token_enforcer = TokenEnforcer(data, parser)
    return LogitsProcessor(token_enforcer)


def _build_regular_tokens_list(tokenizer: Tokenizer, vocab_size: int) -> list[tuple[int, str, bool]]:
    token_0 = tokenizer.encode("0", bos=False, eos=False)[-1]
    regular_tokens = []

    special_token_ids = set(tokenizer.special_tokens.values())
    for token_idx in range(vocab_size):
        if token_idx in special_token_ids:
            continue

        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def _infer_sampling_params(sampling_params: SamplingParams):
    if isinstance(sampling_params.strategy, GreedySamplingStrategy):
        temperature = 0.0
        top_p = 1.0
    elif isinstance(sampling_params.strategy, TopPSamplingStrategy):
        temperature = sampling_params.strategy.temperature or 1.0
        top_p = sampling_params.strategy.top_p or 1.0
    else:
        raise ValueError(f"Unsupported sampling strategy {sampling_params.strategy}")
    return temperature, top_p


class LlamaGenerator:
    def __init__(
        self,
        config: MetaReferenceInferenceConfig,
        model_id: str,
        llama_model: Model,
    ):
        if config.checkpoint_dir and config.checkpoint_dir != "null":
            ckpt_dir = config.checkpoint_dir
        else:
            resolved_model = resolve_model(model_id)
            if resolved_model is None:
                # if the model is not a native llama model, get the default checkpoint_dir based on model id
                ckpt_dir = model_checkpoint_dir(model_id)
            else:
                # if the model is a native llama model, get the default checkpoint_dir based on model core_model_id value
                ckpt_dir = model_checkpoint_dir(resolved_model.descriptor())

        if config.quantization:
            if config.quantization.type == "fp8_mixed":
                quantization_mode = QuantizationMode.fp8_mixed
            elif config.quantization.type == "int4_mixed":
                quantization_mode = QuantizationMode.int4_mixed
            elif config.quantization.type == "bf16":
                quantization_mode = None
            else:
                raise ValueError(f"Unsupported quantization mode {config.quantization}")
        else:
            quantization_mode = None

        cls = Llama4 if llama_model.model_family == ModelFamily.llama4 else Llama3
        self.inner_generator = cls.build(
            ckpt_dir=ckpt_dir,
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
            world_size=config.model_parallel_size or llama_model.pth_file_count,
            quantization_mode=quantization_mode,
        )

        self.tokenizer = self.inner_generator.tokenizer
        self.args = self.inner_generator.args
        self.formatter = self.inner_generator.formatter

    def chat_completion(
        self,
        request: OpenAIChatCompletionRequestWithExtraBody,
        raw_messages: list,
    ):
        """Generate chat completion using OpenAI request format.

        Args:
            request: OpenAI chat completion request
            raw_messages: Pre-converted list of RawMessage objects
        """

        # Determine tool prompt format
        tool_prompt_format = ToolPromptFormat.json if request.tools else ToolPromptFormat.json

        # Prepare sampling params
        sampling_params = SamplingParams()
        if request.temperature is not None or request.top_p is not None:
            sampling_params.strategy = TopPSamplingStrategy(
                temperature=request.temperature if request.temperature is not None else 1.0,
                top_p=request.top_p if request.top_p is not None else 1.0,
            )
        if request.max_tokens:
            sampling_params.max_tokens = request.max_tokens

        max_gen_len = sampling_params.max_tokens
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.args.max_seq_len:
            max_gen_len = self.args.max_seq_len - 1

        temperature, top_p = _infer_sampling_params(sampling_params)

        # Get logits processor for response format
        logits_processor = None
        if request.response_format:
            if isinstance(request.response_format, OpenAIResponseFormatJSONSchema):
                # Extract the actual schema from OpenAIJSONSchema TypedDict
                schema_dict = request.response_format.json_schema.get("schema") or {}
                json_schema_format = JsonSchemaResponseFormat(
                    type=ResponseFormatType.json_schema,
                    json_schema=schema_dict,
                )
                logits_processor = get_logits_processor(self.tokenizer, self.args.vocab_size, json_schema_format)

        # Generate
        yield from self.inner_generator.generate(
            llm_inputs=[self.formatter.encode_dialog_prompt(raw_messages, tool_prompt_format)],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=False,
            echo=False,
            logits_processor=logits_processor,
        )
