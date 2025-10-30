# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import codecs
import io
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

from ..checkpoint import maybe_reshard_state_dict
from ..datatypes import GenerationResult, QuantizationMode
from .args import ModelArgs
from .chat_format import ChatFormat, RawContent, RawMessage
from .datatypes import LLMInput, MaskedEmbedding, TransformerInput
from .model import Transformer
from .tokenizer import Tokenizer

torch.serialization.add_safe_globals([io.BytesIO, codecs.encode])


class Llama4:
    @staticmethod
    def build(
        ckpt_dir: str,
        max_seq_len: int,
        max_batch_size: int,
        world_size: int | None = None,
        quantization_mode: QuantizationMode | None = None,
        seed: int = 1,
    ):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

        if not model_parallel_is_initialized():
            if world_size is None:
                world_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(world_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

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
            **params,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )
        tokenizer = Tokenizer.get_instance()

        # TODO: params.json should always have correct vocab_size
        if model_args.vocab_size == -1:
            model_args.vocab_size = tokenizer.n_words
        assert model_args.vocab_size == tokenizer.n_words, f"{model_args.vocab_size=} vs. {tokenizer.n_words=} mismatch"
        print("Model args:\n", model_args.model_dump_json(indent=2))

        state_dict = maybe_reshard_state_dict(
            ckpt_paths,
            n_kv_heads=model_args.n_kv_heads if model_args.n_kv_heads else model_args.n_heads,
            moe_num_experts=model_args.moe_args.num_experts,
        )
        print("Loaded checkpoint")
        if quantization_mode == QuantizationMode.fp8_mixed or quantization_mode == QuantizationMode.int4_mixed:
            from .quantization.loader import convert_to_quantized_model

            torch.set_default_tensor_type(torch.BFloat16Tensor)
            model = Transformer(model_args)
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=False)
            print("Done...")
            model = convert_to_quantized_model(model, ckpt_dir, quantization_mode)
        else:
            if torch.cuda.is_bf16_supported():
                torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            else:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)

            model = Transformer(model_args)
            print("Loading state dict...")
            model.load_state_dict(state_dict, strict=False)
            print("Done...")
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama4(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer, vision_args=args.vision_args)

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
        if max_gen_len is None or max_gen_len == 0 or max_gen_len >= self.model.args.max_seq_len:
            max_gen_len = self.model.args.max_seq_len - 1

        params = self.model.args

        print_model_input = print_model_input or os.environ.get("LLAMA_MODELS_DEBUG", "0") == "1"
        if print_model_input:
            cprint("Input to model:\n", color="yellow", file=sys.stderr)
            for inp in llm_inputs:
                cprint(self.tokenizer.decode(inp.tokens), color="grey", file=sys.stderr)
        prompt_tokens = [inp.tokens for inp in llm_inputs]

        bsz = len(llm_inputs)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.max_seq_len:
            cprint(f"Out of token budget {max_prompt_len} vs {params.max_seq_len}", color="red", file=sys.stderr)
            return

        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        eos_reached = torch.tensor([False] * bsz, device="cuda")
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

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens, device="cuda")

        prev_pos = 0
        for cur_pos in range(min_prompt_len, total_len):
            image_embedding = None
            if prev_pos == 0 and any(inp.images is not None and len(inp.images) > 0 for inp in llm_inputs):
                image_mask = tokens[:, prev_pos:cur_pos] == self.tokenizer.special_tokens["<|patch|>"]
                image_mask = image_mask.unsqueeze(-1)
                h = self.model.tok_embeddings(tokens[:, prev_pos:cur_pos])

                image_batch = [inp.images if inp.images is not None else [] for inp in llm_inputs]
                image_embedding = MaskedEmbedding(
                    embedding=self.model.vision_embeddings(image_batch, image_mask, h),
                    mask=image_mask,
                )

            xformer_input = TransformerInput(
                tokens=tokens[:, prev_pos:cur_pos],
                tokens_position=prev_pos,
                image_embedding=image_embedding,
            )
            xformer_output = self.model.forward(xformer_input)
            logits = xformer_output.logits
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
        llm_inputs = [self.formatter.encode_content(c) for c in contents]
        for result in self.generate(
            llm_inputs=llm_inputs,
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
        echo: bool = False,
    ) -> Generator[list[GenerationResult], None, None]:
        llm_inputs = [self.formatter.encode_dialog_prompt(messages) for messages in messages_batch]
        for result in self.generate(
            llm_inputs=llm_inputs,
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
