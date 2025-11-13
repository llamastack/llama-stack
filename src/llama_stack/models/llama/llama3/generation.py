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

import json
import os
import sys
import time
from collections.abc import Callable, Generator
from pathlib import Path

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from termcolor import cprint

from llama_stack.models.llama.datatypes import ToolPromptFormat

from ..checkpoint import maybe_reshard_state_dict
from ..datatypes import GenerationResult, QuantizationMode, RawContent, RawMessage
from .args import ModelArgs
from .chat_format import ChatFormat, LLMInput
from .model import Transformer
from .multimodal.model import CrossAttentionTransformer
from .tokenizer import Tokenizer


class Llama3:
    @staticmethod
    def build(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        world_size: int | None = None,
        quantization_mode: QuantizationMode | None = None,
        seed: int = 1,
        device: str = "cuda",
    ):
        device = torch.device(device)
        if (
            device.type == "cuda"
            and not torch.cuda.is_available()
            or device.type == "xpu"
            and not torch.xpu.is_available()
        ):
            raise RuntimeError(f"PyTorch backend for {device.type} device type is not available")

        if not torch.distributed.is_initialized():
            if device.type == "cuda":
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("gloo")

        if not model_parallel_is_initialized():
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(world_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        elif device.type == "xpu":
            torch.xpu.set_device(local_rank)

        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()

        ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(ckpt_paths) > 0, f"no checkpoint files found in {ckpt_dir}"
        print(f"Loading a checkpoint (shards={len(ckpt_paths)}, current-mp-size={world_size})")
        with open(Path(ckpt_dir) / "params.json") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer.get_instance()

        state_dict = maybe_reshard_state_dict(
            ckpt_paths,
            n_kv_heads=model_args.n_kv_heads if model_args.n_kv_heads else model_args.n_heads,
        )

        assert model_args.vocab_size == tokenizer.n_words

        def build_model():
            if model_args.vision_chunk_size > 0:
                model = CrossAttentionTransformer(model_args)
                model.setup_cache(model_args.max_batch_size, device=device, dtype=torch.get_default_dtype())
            else:
                model = Transformer(model_args)
            return model

        if quantization_mode == QuantizationMode.fp8_mixed or quantization_mode == QuantizationMode.int4_mixed:
            from .quantization.loader import convert_to_quantized_model

            torch.set_default_tensor_type(torch.BFloat16Tensor)
            model = build_model()
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=False)
            print("Done...")
            model = convert_to_quantized_model(model, ckpt_dir, quantization_mode, device=device)
            torch.set_default_device(device)
        else:
            print(f"Setting default device to {device}")
            if device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
                else:
                    torch.set_default_tensor_type(torch.cuda.Float16Tensor)
            elif device.type == "xpu":
                if torch.xpu.is_bf16_supported():
                    torch.set_default_tensor_type(torch.xpu.BFloat16Tensor)
                else:
                    torch.set_default_tensor_type(torch.xpu.Float16Tensor)

            model = build_model()
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=True)
            model.to(device)
            print("Done...")

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama3(model, tokenizer, model_args)

    def __init__(
        self,
        model: Transformer | CrossAttentionTransformer,
        tokenizer: Tokenizer,
        args: ModelArgs,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        llm_inputs: list[LLMInput],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        echo: bool = False,
        print_model_input: bool = False,
        logits_processor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> Generator[list[GenerationResult], None, None]:
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.args.max_seq_len:
            max_gen_len = self.args.max_seq_len - 1
        params = self.model.params

        print_model_input = print_model_input or os.environ.get("LLAMA_MODELS_DEBUG", "0") == "1"
        if print_model_input:
            for inp in llm_inputs:
                tokens_to_print = [self.formatter.vision_token if t == 128256 else t for t in inp.tokens]
                cprint(
                    "Input to model:\n" + self.tokenizer.decode(tokens_to_print) + "\n",
                    "red",
                    file=sys.stderr,
                )
        prompt_tokens = [inp.tokens for inp in llm_inputs]

        bsz = len(llm_inputs)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.max_seq_len:
            cprint(
                f"Out of token budget {max_prompt_len} vs {params.max_seq_len}",
                color="red",
                file=sys.stderr,
            )
            return

        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        is_vision = not isinstance(self.model, Transformer)
        if is_vision:
            images = [inp.vision.images if inp.vision is not None else [] for inp in llm_inputs]
            mask = [inp.vision.mask if inp.vision is not None else [] for inp in llm_inputs]

            xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
                batch_images=images,
                batch_masks=mask,
                total_len=total_len,
                device=tokens.device,
            )

        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id

        if echo:
            for i in range(max_prompt_len):
                results = []
                for j, t in enumerate(tokens[:, i]):
                    results.append(
                        GenerationResult(
                            token=t.item(),
                            text=self.tokenizer.decode([t.item()]),
                            source="input",
                            logprobs=(token_logprobs[j, i : i + 1].tolist() if logprobs else None),
                            batch_idx=j,
                            finished=False,
                            ignore_token=t.item() == pad_id,
                        )
                    )
                yield results

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)

        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            if is_vision:
                position_ids = torch.arange(prev_pos, cur_pos, dtype=torch.long)
                text_only_inference = all(inp.vision is None for inp in llm_inputs)
                logits = self.model.forward(
                    position_ids,
                    tokens,
                    cross_attention_masks,
                    full_text_row_masked_out_mask,
                    xattn_caches,
                    text_only_inference,
                )
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if logits_processor is not None:
                logits = logits_processor(tokens[:, :cur_pos], logits)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]
            if is_vision:
                # the logits space (num_classes) is designed to never contain a media_token
                # however our input token stream does contain them. we need to nuke them here
                # or else the CUDA kernels will crash with an illegal memory access
                vision_tokens = [self.tokenizer.special_tokens["<|image|>"], 128256]
                masks = [target.eq(t) for t in vision_tokens]
                if len(masks) > 1:
                    mask = torch.logical_or(*masks)
                else:
                    mask = masks[0]
                target[mask] = 0

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=target,
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            results = []
            for idx, t in enumerate(next_token):
                results.append(
                    GenerationResult(
                        token=t.item(),
                        text=self.tokenizer.decode([t.item()]),
                        source="output",
                        logprobs=(token_logprobs[idx, cur_pos : cur_pos + 1].tolist() if logprobs else None),
                        batch_idx=idx,
                        finished=eos_reached[idx].item(),
                        ignore_token=cur_pos < len(prompt_tokens[idx]),
                    )
                )
            yield results

            prev_pos = cur_pos
            if all(eos_reached):
                break

    def completion(
        self,
        contents: list[RawContent],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Generator[list[GenerationResult], None, None]:
        model_inputs = [self.formatter.encode_content(c) for c in contents]
        for result in self.generate(
            model_inputs=model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        ):
            yield result
            if all(r.finished for r in result):
                break

    def chat_completion(
        self,
        messages_batch: list[list[RawMessage]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: int | None = None,
        logprobs: bool = False,
        tool_prompt_format: ToolPromptFormat = ToolPromptFormat.json,
        echo: bool = False,
    ) -> Generator[list[GenerationResult], None, None]:
        model_inputs = [self.formatter.encode_dialog_prompt(messages) for messages in messages_batch]
        for result in self.generate(
            model_inputs=model_inputs,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
            logprobs=logprobs,
            echo=echo,
        ):
            yield result
            if all(r.finished for r in result):
                break


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
