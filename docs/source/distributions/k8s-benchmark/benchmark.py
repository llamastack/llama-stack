# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Simple benchmark script for Llama Stack with OpenAI API compatibility.
"""

import argparse
import asyncio
import os
import random
import statistics
import time
from typing import Tuple
import aiohttp


class BenchmarkStats:
    def __init__(self):
        self.response_times = []
        self.ttft_times = []
        self.chunks_received = []
        self.errors = []
        self.success_count = 0
        self.total_requests = 0
        self.concurrent_users = 0
        self.start_time = None
        self.end_time = None
        self._lock = asyncio.Lock()

    async def add_result(self, response_time: float, chunks: int, ttft: float = None, error: str = None):
        async with self._lock:
            self.total_requests += 1
            if error:
                self.errors.append(error)
            else:
                self.success_count += 1
                self.response_times.append(response_time)
                self.chunks_received.append(chunks)
                if ttft is not None:
                    self.ttft_times.append(ttft)

    def print_summary(self):
        if not self.response_times:
            print("No successful requests to report")
            if self.errors:
                print(f"Total errors: {len(self.errors)}")
                print("First 5 errors:")
                for error in self.errors[:5]:
                    print(f"  {error}")
            return

        total_time = self.end_time - self.start_time
        success_rate = (self.success_count / self.total_requests) * 100
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS")
        
        print(f"\nResponse Time Statistics:")
        print(f"  Mean: {statistics.mean(self.response_times):.3f}s")
        print(f"  Median: {statistics.median(self.response_times):.3f}s")
        print(f"  Min: {min(self.response_times):.3f}s")
        print(f"  Max: {max(self.response_times):.3f}s")
        
        if len(self.response_times) > 1:
            print(f"  Std Dev: {statistics.stdev(self.response_times):.3f}s")
            
        percentiles = [50, 90, 95, 99]
        sorted_times = sorted(self.response_times)
        print(f"\nPercentiles:")
        for p in percentiles:
            idx = int(len(sorted_times) * p / 100) - 1
            idx = max(0, min(idx, len(sorted_times) - 1))
            print(f"  P{p}: {sorted_times[idx]:.3f}s")
            
        if self.ttft_times:
            print(f"\nTime to First Token (TTFT) Statistics:")
            print(f"  Mean: {statistics.mean(self.ttft_times):.3f}s")
            print(f"  Median: {statistics.median(self.ttft_times):.3f}s")
            print(f"  Min: {min(self.ttft_times):.3f}s")
            print(f"  Max: {max(self.ttft_times):.3f}s")
            
            if len(self.ttft_times) > 1:
                print(f"  Std Dev: {statistics.stdev(self.ttft_times):.3f}s")
                
            sorted_ttft = sorted(self.ttft_times)
            print(f"\nTTFT Percentiles:")
            for p in percentiles:
                idx = int(len(sorted_ttft) * p / 100) - 1
                idx = max(0, min(idx, len(sorted_ttft) - 1))
                print(f"  P{p}: {sorted_ttft[idx]:.3f}s")
            
        if self.chunks_received:
            print(f"\nStreaming Statistics:")
            print(f"  Mean chunks per response: {statistics.mean(self.chunks_received):.1f}")
            print(f"  Total chunks received: {sum(self.chunks_received)}")
        
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Concurrent users: {self.concurrent_users}")
        print(f"Total requests: {self.total_requests}")
        print(f"Successful requests: {self.success_count}")
        print(f"Failed requests: {len(self.errors)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Requests per second: {self.success_count / total_time:.2f}")
        
        if self.errors:
            print(f"\nErrors (showing first 5):")
            for error in self.errors[:5]:
                print(f"  {error}")


class LlamaStackBenchmark:
    def __init__(self, base_url: str, model_id: str):
        self.base_url = base_url.rstrip('/')
        self.model_id = model_id
        self.headers = {"Content-Type": "application/json"}
        self.test_messages = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "What is the capital of France?"}],
            [{"role": "user", "content": "Explain quantum physics in simple terms."}],
            [{"role": "user", "content": "Write a short story about a robot learning to paint."}],
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI..."},
                {"role": "user", "content": "Can you give me a practical example?"}
            ]
        ]


    async def make_async_streaming_request(self) -> Tuple[float, int, float | None, str | None]:
        """Make a single async streaming chat completion request."""
        messages = random.choice(self.test_messages)
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "max_tokens": 100
        }
        
        start_time = time.time()
        chunks_received = 0
        ttft = None
        error = None
        
        session = aiohttp.ClientSession()
        
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                chunks_received += 1
                                if ttft is None:
                                    ttft = time.time() - start_time
                                if line_str == 'data: [DONE]':
                                    break
                    
                    if chunks_received == 0:
                        error = "No streaming chunks received"
                else:
                    text = await response.text()
                    error = f"HTTP {response.status}: {text[:100]}"
                    
        except Exception as e:
            error = f"Request error: {str(e)}"
        finally:
            await session.close()
            
        response_time = time.time() - start_time
        return response_time, chunks_received, ttft, error


    async def run_benchmark(self, duration: int, concurrent_users: int) -> BenchmarkStats:
        """Run benchmark using async requests for specified duration."""
        stats = BenchmarkStats()
        stats.concurrent_users = concurrent_users
        stats.start_time = time.time()
        
        print(f"Starting benchmark: {duration}s duration, {concurrent_users} concurrent users")
        print(f"Target URL: {self.base_url}/chat/completions")
        print(f"Model: {self.model_id}")
        
        connector = aiohttp.TCPConnector(limit=concurrent_users)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            async def worker(worker_id: int):
                """Worker that sends requests sequentially until canceled."""
                request_count = 0
                while True:
                    try:
                        response_time, chunks, ttft, error = await self.make_async_streaming_request()
                        await stats.add_result(response_time, chunks, ttft, error)
                        request_count += 1
                        
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        await stats.add_result(0, 0, None, f"Worker {worker_id} error: {str(e)}")
            
            # Progress reporting task
            async def progress_reporter():
                last_report_time = time.time()
                while True:
                    try:
                        await asyncio.sleep(1)  # Report every second
                        if time.time() >= last_report_time + 10:  # Report every 10 seconds
                            elapsed = time.time() - stats.start_time
                            print(f"Completed: {stats.total_requests} requests in {elapsed:.1f}s, RPS: {stats.total_requests / elapsed:.1f}")
                            last_report_time = time.time()
                    except asyncio.CancelledError:
                        break
            
            # Spawn concurrent workers
            tasks = [asyncio.create_task(worker(i)) for i in range(concurrent_users)]
            progress_task = asyncio.create_task(progress_reporter())
            tasks.append(progress_task)
            
            # Wait for duration then cancel all tasks
            await asyncio.sleep(duration)
            
            for task in tasks:
                task.cancel()
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        stats.end_time = time.time()
        return stats


def main():
    parser = argparse.ArgumentParser(description="Llama Stack Benchmark Tool")
    parser.add_argument("--base-url", default=os.getenv("BENCHMARK_BASE_URL", "http://localhost:8000/v1/openai/v1"),
                       help="Base URL for the API (default: http://localhost:8000/v1/openai/v1)")
    parser.add_argument("--model", default=os.getenv("INFERENCE_MODEL", "test-model"),
                       help="Model ID to use for requests")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration in seconds to run benchmark (default: 60)")
    parser.add_argument("--concurrent", type=int, default=10,
                       help="Number of concurrent users (default: 10)")
    
    args = parser.parse_args()
    
    benchmark = LlamaStackBenchmark(args.base_url, args.model)
    
    try:
        stats = asyncio.run(benchmark.run_benchmark(args.duration, args.concurrent))
        stats.print_summary()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == "__main__":
    main()
