#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Benchmark: guardrail batching performance against a running OGX server.

Makes streaming Responses API requests with guardrails enabled and measures
wall-clock time, time-to-first-token, and token throughput.

Prerequisites:
    1. Start Ollama:  ollama serve
    2. Start OGX with safety enabled:
         SAFETY_MODEL=llama-guard3:1b uv run ogx run starter

    To A/B test the batching optimization, restart the server with:
         GUARDRAIL_BATCH_CHARS=1   → per-token checking (before)
         GUARDRAIL_BATCH_CHARS=200 → batched checking (after, default)

Usage:
    uv run python scripts/benchmark_guardrail_batching.py
    uv run python scripts/benchmark_guardrail_batching.py --model ollama/llama3.2:3b --runs 5
    uv run python scripts/benchmark_guardrail_batching.py --no-guardrails  # baseline without guardrails
    uv run python scripts/benchmark_guardrail_batching.py --help
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass

import httpx

DEFAULT_PROMPT = "Explain the concept of recursion in programming. Give a short example."


@dataclass
class RunResult:
    elapsed_s: float
    ttft_s: float
    token_count: int
    event_count: int
    status: str

    @property
    def tokens_per_s(self) -> float:
        if self.elapsed_s == 0:
            return 0
        return self.token_count / self.elapsed_s


async def stream_response(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    guardrail_id: str | None,
) -> RunResult:
    body: dict = {
        "model": model,
        "input": prompt,
        "stream": True,
        "max_output_tokens": max_tokens,
    }
    if guardrail_id:
        body["guardrails"] = [guardrail_id]

    t_start = time.perf_counter()
    ttft = 0.0
    token_count = 0
    event_count = 0
    status = "completed"

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream(
            "POST",
            f"{base_url}/v1/responses",
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                text = await resp.aread()
                return RunResult(
                    elapsed_s=time.perf_counter() - t_start,
                    ttft_s=0,
                    token_count=0,
                    event_count=0,
                    status=f"error:{resp.status_code} {text.decode()[:200]}",
                )

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event_count += 1
                payload = json.loads(line[6:])
                event_type = payload.get("type", "")

                if event_type == "response.output_text.delta":
                    token_count += 1
                    if ttft == 0:
                        ttft = time.perf_counter() - t_start
                elif event_type == "response.failed":
                    status = "failed"
                elif event_type == "response.incomplete":
                    status = "incomplete"

    elapsed = time.perf_counter() - t_start
    return RunResult(
        elapsed_s=elapsed,
        ttft_s=ttft,
        token_count=token_count,
        event_count=event_count,
        status=status,
    )


def _print_result(label: str, results: list[RunResult]) -> None:
    ok = [r for r in results if r.status == "completed"]
    if not ok:
        print(f"\n  {label}: all runs failed — {results[0].status}")
        return

    avg_elapsed = sum(r.elapsed_s for r in ok) / len(ok)
    avg_ttft = sum(r.ttft_s for r in ok) / len(ok)
    avg_tokens = sum(r.token_count for r in ok) / len(ok)
    avg_tps = sum(r.tokens_per_s for r in ok) / len(ok)

    print(f"\n  {label} ({len(ok)}/{len(results)} runs ok)")
    print(f"    tokens:          {avg_tokens:>8.0f}")
    print(f"    wall-clock:      {avg_elapsed:>8.3f}s")
    print(f"    TTFT:            {avg_ttft:>8.3f}s")
    print(f"    tokens/s:        {avg_tps:>8.1f}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark guardrail batching against a running OGX server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
A/B testing:
  1. Start server: GUARDRAIL_BATCH_CHARS=1 SAFETY_MODEL=llama-guard3:1b uv run ogx run starter
     Run: uv run python scripts/benchmark_guardrail_batching.py
  2. Restart:      SAFETY_MODEL=llama-guard3:1b uv run ogx run starter
     Run: uv run python scripts/benchmark_guardrail_batching.py
  3. Compare the numbers.
""",
    )
    parser.add_argument("--base-url", default="http://localhost:8321", help="OGX server URL")
    parser.add_argument("--model", default="openai/gpt-4.1-nano", help="Model to use")
    parser.add_argument("--guardrail", default="llama-guard", help="Guardrail shield ID")
    parser.add_argument("--no-guardrails", action="store_true", help="Run without guardrails (baseline)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens per request")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    args = parser.parse_args()

    guardrail_id = None if args.no_guardrails else args.guardrail

    print("Guardrail Batching Benchmark")
    print(f"  Server:     {args.base_url}")
    print(f"  Model:      {args.model}")
    print(f"  Guardrail:  {guardrail_id or '(none)'}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Runs:       {args.runs}")

    # Warmup run
    print("\n  Warming up...", end="", flush=True)
    warmup = await stream_response(args.base_url, args.model, "Say hi", 16, None)
    if warmup.status != "completed":
        print(f"\n  ERROR: warmup failed — {warmup.status}")
        print("  Is the OGX server running? Try: SAFETY_MODEL=llama-guard3:1b uv run ogx run starter")
        return
    print(" done")

    # Benchmark runs
    results: list[RunResult] = []
    for i in range(args.runs):
        print(f"  Run {i + 1}/{args.runs}...", end="", flush=True)
        r = await stream_response(args.base_url, args.model, args.prompt, args.max_tokens, guardrail_id)
        results.append(r)
        print(f" {r.token_count} tokens, {r.elapsed_s:.3f}s ({r.status})")

    _print_result(f"model={args.model}, guardrail={guardrail_id or 'none'}", results)
    print()


if __name__ == "__main__":
    asyncio.run(main())
