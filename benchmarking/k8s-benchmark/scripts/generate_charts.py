#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# /// script
# dependencies = [
#   "matplotlib",
# ]
# ///
"""
Script to generate benchmark charts from guidellm JSON results.
Plots TTFT and ITL metrics against constant@x values.
"""

import glob
import os
import re

import matplotlib.pyplot as plt


def extract_setup_name(filename: str) -> str:
    """Extract setup name from filename and format legend appropriately."""
    # Pattern: guidellm-benchmark-{target}-{stack_replicas}-{vllm_replicas}-{timestamp}.txt
    basename = os.path.basename(filename)
    match = re.search(r"guidellm-benchmark-([^-]+)-(\d+)-(\d+)-(\d+)-(\d+)\.txt", basename)
    if match:
        target = match.group(1)
        stack_replicas = match.group(2)
        vllm_replicas = match.group(3)
        date = match.group(4)
        time = match.group(5)

        if target == "vllm":
            return f"vllm-replicas{vllm_replicas}-{date}-{time}"
        else:
            return f"stack-replicas{stack_replicas};vllm-replicas{vllm_replicas}-{date}-{time}"
    return basename.replace("guidellm-benchmark-", "").replace(".txt", "")


def parse_txt_file(filepath: str) -> list[tuple[float, float, float, float, str]]:
    """
    Parse a text benchmark file and extract constant@x, TTFT, ITL, and request latency data.
    Returns list of (constant_rate, ttft_mean, itl_mean, req_latency_mean, setup_name) tuples.
    """
    setup_name = extract_setup_name(filepath)
    data_points = []

    try:
        with open(filepath) as f:
            content = f.read()

        # Find the benchmark stats table
        lines = content.split("\n")
        in_stats_table = False
        header_lines_seen = 0

        for line in lines:
            line_stripped = line.strip()

            # Look for the start of the stats table
            if "Benchmarks Stats:" in line:
                in_stats_table = True
                continue

            if in_stats_table:
                # Skip the first few separator/header lines
                if line_stripped.startswith("=") or line_stripped.startswith("-"):
                    header_lines_seen += 1
                    if header_lines_seen >= 3:  # After seeing multiple header lines, look for constant@ data
                        if line_stripped.startswith("=") and "constant@" not in line_stripped:
                            break
                    continue

            # Parse constant@ lines in the stats table (may have leading spaces)
            if in_stats_table and "constant@" in line:
                parts = [part.strip() for part in line.split("|")]

                if len(parts) >= 12:  # Make sure we have enough columns
                    try:
                        # Extract constant rate from benchmark name (e.g., constant@1.12 -> 1.12)
                        constant_match = re.search(r"constant@([\d.]+)", parts[0])
                        if not constant_match:
                            continue
                        constant_rate = float(constant_match.group(1))

                        # Extract request latency, TTFT and ITL means from the table
                        # Table structure: Benchmark | Per Second | Concurrency | ... | Req Latency mean | ... | TTFT mean | ... | ITL mean | ...
                        # Index positions: 0        1             2             ...   5                  ...   8           ...   11         ...
                        req_latency_mean = float(parts[5])  # Request latency mean column (in seconds, convert to ms)
                        ttft_mean = float(parts[8])  # TTFT mean column
                        itl_mean = float(parts[11])  # ITL mean column

                        data_points.append((constant_rate, ttft_mean, itl_mean, req_latency_mean * 1000, setup_name))

                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line '{line}' in {filepath}: {e}")
                        continue

    except (OSError, FileNotFoundError) as e:
        print(f"Error reading {filepath}: {e}")

    return data_points


def generate_charts(benchmark_dir: str = "results"):
    """Generate TTFT and ITL charts from benchmark text files."""
    # Find all text result files instead of JSON
    txt_pattern = os.path.join(benchmark_dir, "guidellm-benchmark-*.txt")
    txt_files = glob.glob(txt_pattern)

    if not txt_files:
        print(f"No text files found matching pattern: {txt_pattern}")
        return

    print(f"Found {len(txt_files)} text files")

    # Parse all files and collect data
    all_data = {}  # setup_name -> [(constant_rate, ttft, itl), ...]

    for txt_file in txt_files:
        print(f"Processing {txt_file}")
        data_points = parse_txt_file(txt_file)

        for constant_rate, ttft, itl, req_latency, setup_name in data_points:
            if setup_name not in all_data:
                all_data[setup_name] = []
            all_data[setup_name].append((constant_rate, ttft, itl, req_latency))

    if not all_data:
        print("No data found to plot")
        return

    # Sort data points by constant rate for each setup
    for setup_name in all_data:
        all_data[setup_name].sort(key=lambda x: x[0])  # Sort by constant_rate

    # Group setups by vLLM replica number dynamically
    replica_groups = {}  # vllm_replica_count -> {setup_name: points}

    for setup_name, points in all_data.items():
        # Extract vLLM replica number from setup name
        # Expected formats: "stack-replicas{X};vllm-replicas{Y}" or "vllm-replicas{Y}"
        vllm_match = re.search(r"vllm-replicas(\d+)", setup_name)
        if vllm_match:
            vllm_replica_num = int(vllm_match.group(1))
            if vllm_replica_num not in replica_groups:
                replica_groups[vllm_replica_num] = {}
            replica_groups[vllm_replica_num][setup_name] = points
        else:
            print(f"Warning: Could not extract vLLM replica count from setup name: {setup_name}")

    def create_charts(data_dict, prefix, title_prefix):
        """Create TTFT and ITL charts for a given data dictionary."""
        if not data_dict:
            print(f"No data found for {prefix}")
            return

        # Create TTFT chart
        plt.figure(figsize=(14, 8))
        for setup_name, points in data_dict.items():
            if not points:
                continue

            constant_rates = [p[0] for p in points]
            ttft_values = [p[1] for p in points]

            plt.plot(constant_rates, ttft_values, marker="o", label=setup_name, linewidth=2, markersize=6)

        plt.xlabel("Concurrency (req/s)", fontsize=12)
        plt.ylabel("Time to First Token (ms)", fontsize=12)
        plt.title(f"{title_prefix} TTFT vs Concurrency", fontsize=14, fontweight="bold")
        plt.ylim(bottom=0)  # Start y-axis from 0
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        ttft_filename = os.path.join(benchmark_dir, f"{prefix}_ttft_vs_concurrency.png")
        plt.savefig(ttft_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"TTFT chart saved to {ttft_filename}")

        # Create ITL chart
        plt.figure(figsize=(14, 8))
        for setup_name, points in data_dict.items():
            if not points:
                continue

            constant_rates = [p[0] for p in points]
            itl_values = [p[2] for p in points]

            plt.plot(constant_rates, itl_values, marker="o", label=setup_name, linewidth=2, markersize=6)

        plt.xlabel("Concurrency (req/s)", fontsize=12)
        plt.ylabel("Inter Token Latency (ms)", fontsize=12)
        plt.title(f"{title_prefix} ITL vs Concurrency", fontsize=14, fontweight="bold")
        plt.ylim(bottom=0)  # Start y-axis from 0
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        itl_filename = os.path.join(benchmark_dir, f"{prefix}_itl_vs_concurrency.png")
        plt.savefig(itl_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"ITL chart saved to {itl_filename}")

        # Create Request Latency chart
        plt.figure(figsize=(14, 8))
        for setup_name, points in data_dict.items():
            if not points:
                continue

            constant_rates = [p[0] for p in points]
            req_latency_values = [p[3] for p in points]

            plt.plot(constant_rates, req_latency_values, marker="o", label=setup_name, linewidth=2, markersize=6)

        plt.xlabel("Concurrency (req/s)", fontsize=12)
        plt.ylabel("Request Latency (ms)", fontsize=12)
        plt.title(f"{title_prefix} Request Latency vs Concurrency", fontsize=14, fontweight="bold")
        plt.ylim(bottom=0)  # Start y-axis from 0
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        req_latency_filename = os.path.join(benchmark_dir, f"{prefix}_req_latency_vs_concurrency.png")
        plt.savefig(req_latency_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Request Latency chart saved to {req_latency_filename}")

    # Print grouping information
    for replica_count, data_dict in replica_groups.items():
        print(f"vLLM Replica {replica_count} setups: {list(data_dict.keys())}")

    # Create separate charts for each replica group
    for replica_count, data_dict in replica_groups.items():
        prefix = f"vllm_replica{replica_count}"
        title = f"vLLM Replicas={replica_count}"
        create_charts(data_dict, prefix, title)

    # Print summary
    print("\nSummary:")
    for setup_name, points in all_data.items():
        print(f"{setup_name}: {len(points)} data points")


if __name__ == "__main__":
    generate_charts()
