# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from termcolor import cprint

from llama_stack.cli.subcommand import Subcommand
from llama_stack.models.llama.sku_list import LlamaDownloadInfo
from llama_stack.models.llama.sku_types import Model


class Download(Subcommand):
    """Llama cli for downloading llama toolchain assets"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "download",
            prog="llama download",
            description="Download a model from llama.meta.com or Hugging Face Hub",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        setup_download_parser(self.parser)


def setup_download_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=["meta", "huggingface"],
        default="meta",
    )
    parser.add_argument(
        "--model-id",
        required=False,
        help="See `llama model list` or `llama model list --show-all` for the list of available models. Specify multiple model IDs with commas, e.g. --model-id Llama3.2-1B,Llama3.2-3B",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=False,
        default=None,
        help="Hugging Face API token. Needed for gated models like llama2/3. Will also try to read environment variable `HF_TOKEN` as default.",
    )
    parser.add_argument(
        "--meta-url",
        type=str,
        required=False,
        help="For source=meta, URL obtained from llama.meta.com after accepting license terms",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        required=False,
        default=3,
        help="Maximum number of concurrent downloads",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        required=False,
        default="*.safetensors",
        help="""For source=huggingface, files matching any of the patterns are not downloaded. Defaults to ignoring
safetensors files to avoid downloading duplicate weights.
""",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        help="For source=meta, you can download models from a manifest file containing a file => URL mapping",
        required=False,
    )
    parser.set_defaults(func=partial(run_download_cmd, parser=parser))


@dataclass
class DownloadTask:
    url: str
    output_file: str
    total_size: int = 0
    downloaded_size: int = 0
    task_id: int | None = None
    retries: int = 0
    max_retries: int = 3


class DownloadError(Exception):
    pass


class CustomTransferSpeedColumn(TransferSpeedColumn):
    def render(self, task):
        if task.finished:
            return "-"
        return super().render(task)


class ParallelDownloader:
    def __init__(
        self,
        max_concurrent_downloads: int = 3,
        buffer_size: int = 1024 * 1024,
        timeout: int = 30,
    ):
        self.max_concurrent_downloads = max_concurrent_downloads
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.console = Console()
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            DownloadColumn(),
            CustomTransferSpeedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.client_options = {
            "timeout": httpx.Timeout(timeout),
            "follow_redirects": True,
        }

    async def retry_with_exponential_backoff(self, task: DownloadTask, func, *args, **kwargs):
        last_exception = None
        for attempt in range(task.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < task.max_retries - 1:
                    wait_time = min(30, 2**attempt)  # Cap at 30 seconds
                    self.console.print(
                        f"[yellow]Attempt {attempt + 1}/{task.max_retries} failed, "
                        f"retrying in {wait_time} seconds: {str(e)}[/yellow]"
                    )
                    await asyncio.sleep(wait_time)
                    continue
        raise last_exception

    async def get_file_info(self, client: httpx.AsyncClient, task: DownloadTask) -> None:
        if task.total_size > 0:
            self.progress.update(task.task_id, total=task.total_size)
            return

        async def _get_info():
            response = await client.head(task.url, headers={"Accept-Encoding": "identity"}, **self.client_options)
            response.raise_for_status()
            return response

        try:
            response = await self.retry_with_exponential_backoff(task, _get_info)

            task.url = str(response.url)
            task.total_size = int(response.headers.get("Content-Length", 0))

            if task.total_size == 0:
                raise DownloadError(
                    f"Unable to determine file size for {task.output_file}. "
                    "The server might not support range requests."
                )

            # Update the progress bar's total size once we know it
            if task.task_id is not None:
                self.progress.update(task.task_id, total=task.total_size)

        except httpx.HTTPError as e:
            self.console.print(f"[red]Error getting file info: {str(e)}[/red]")
            raise

    def verify_file_integrity(self, task: DownloadTask) -> bool:
        if not os.path.exists(task.output_file):
            return False
        return os.path.getsize(task.output_file) == task.total_size

    async def download_chunk(self, client: httpx.AsyncClient, task: DownloadTask, start: int, end: int) -> None:
        async def _download_chunk():
            headers = {"Range": f"bytes={start}-{end}"}
            async with client.stream("GET", task.url, headers=headers, **self.client_options) as response:
                response.raise_for_status()

                with open(task.output_file, "ab") as file:
                    file.seek(start)
                    async for chunk in response.aiter_bytes(self.buffer_size):
                        file.write(chunk)
                        task.downloaded_size += len(chunk)
                        self.progress.update(
                            task.task_id,
                            completed=task.downloaded_size,
                        )

        try:
            await self.retry_with_exponential_backoff(task, _download_chunk)
        except Exception as e:
            raise DownloadError(
                f"Failed to download chunk {start}-{end} after {task.max_retries} attempts: {str(e)}"
            ) from e

    async def prepare_download(self, task: DownloadTask) -> None:
        output_dir = os.path.dirname(task.output_file)
        os.makedirs(output_dir, exist_ok=True)

        if os.path.exists(task.output_file):
            task.downloaded_size = os.path.getsize(task.output_file)

    async def download_file(self, task: DownloadTask) -> None:
        try:
            async with httpx.AsyncClient(**self.client_options) as client:
                await self.get_file_info(client, task)

                # Check if file is already downloaded
                if os.path.exists(task.output_file):
                    if self.verify_file_integrity(task):
                        self.console.print(f"[green]Already downloaded {task.output_file}[/green]")
                        self.progress.update(task.task_id, completed=task.total_size)
                        return

                await self.prepare_download(task)

                try:
                    # Split the remaining download into chunks
                    chunk_size = 27_000_000_000  # Cloudfront max chunk size
                    chunks = []

                    current_pos = task.downloaded_size
                    while current_pos < task.total_size:
                        chunk_end = min(current_pos + chunk_size - 1, task.total_size - 1)
                        chunks.append((current_pos, chunk_end))
                        current_pos = chunk_end + 1

                    # Download chunks in sequence
                    for chunk_start, chunk_end in chunks:
                        await self.download_chunk(client, task, chunk_start, chunk_end)

                except Exception as e:
                    raise DownloadError(f"Download failed: {str(e)}") from e

        except Exception as e:
            self.progress.update(task.task_id, description=f"[red]Failed: {task.output_file}[/red]")
            raise DownloadError(f"Download failed for {task.output_file}: {str(e)}") from e

    def has_disk_space(self, tasks: list[DownloadTask]) -> bool:
        try:
            total_remaining_size = sum(task.total_size - task.downloaded_size for task in tasks)
            dir_path = os.path.dirname(os.path.abspath(tasks[0].output_file))
            free_space = shutil.disk_usage(dir_path).free

            # Add 10% buffer for safety
            required_space = int(total_remaining_size * 1.1)

            if free_space < required_space:
                self.console.print(
                    f"[red]Not enough disk space. Required: {required_space // (1024 * 1024)} MB, "
                    f"Available: {free_space // (1024 * 1024)} MB[/red]"
                )
                return False
            return True

        except Exception as e:
            raise DownloadError(f"Failed to check disk space: {str(e)}") from e

    async def download_all(self, tasks: list[DownloadTask]) -> None:
        if not tasks:
            raise ValueError("No download tasks provided")

        if not os.environ.get("LLAMA_DOWNLOAD_NO_SPACE_CHECK") and not self.has_disk_space(tasks):
            raise DownloadError("Insufficient disk space for downloads")

        failed_tasks = []

        with self.progress:
            for task in tasks:
                desc = f"Downloading {Path(task.output_file).name}"
                task.task_id = self.progress.add_task(desc, total=task.total_size, completed=task.downloaded_size)

            semaphore = asyncio.Semaphore(self.max_concurrent_downloads)

            async def download_with_semaphore(task: DownloadTask):
                async with semaphore:
                    try:
                        await self.download_file(task)
                    except Exception as e:
                        failed_tasks.append((task, str(e)))

            await asyncio.gather(*(download_with_semaphore(task) for task in tasks))

        if failed_tasks:
            self.console.print("\n[red]Some downloads failed:[/red]")
            for task, error in failed_tasks:
                self.console.print(f"[red]- {Path(task.output_file).name}: {error}[/red]")
            raise DownloadError(f"{len(failed_tasks)} downloads failed")


def _hf_download(
    model: "Model",
    hf_token: str,
    ignore_patterns: str,
    parser: argparse.ArgumentParser,
):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    from llama_stack.core.utils.model_utils import model_local_dir

    repo_id = model.huggingface_repo
    if repo_id is None:
        raise ValueError(f"No repo id found for model {model.descriptor()}")

    output_dir = model_local_dir(model.descriptor())
    os.makedirs(output_dir, exist_ok=True)
    try:
        true_output_dir = snapshot_download(
            repo_id,
            local_dir=output_dir,
            ignore_patterns=ignore_patterns,
            token=hf_token,
            library_name="llama-stack",
        )
    except GatedRepoError:
        parser.error(
            "It looks like you are trying to access a gated repository. Please ensure you "
            "have access to the repository and have provided the proper Hugging Face API token "
            "using the option `--hf-token` or by running `huggingface-cli login`."
            "You can find your token by visiting https://huggingface.co/settings/tokens"
        )
    except RepositoryNotFoundError:
        parser.error(f"Repository '{repo_id}' not found on the Hugging Face Hub or incorrect Hugging Face token.")
    except Exception as e:
        parser.error(e)

    print(f"\nSuccessfully downloaded model to {true_output_dir}")


def _meta_download(
    model: "Model",
    model_id: str,
    meta_url: str,
    info: "LlamaDownloadInfo",
    max_concurrent_downloads: int,
):
    from llama_stack.core.utils.model_utils import model_local_dir

    output_dir = Path(model_local_dir(model.descriptor()))
    os.makedirs(output_dir, exist_ok=True)

    # Create download tasks for each file
    tasks = []
    for f in info.files:
        output_file = str(output_dir / f)
        url = meta_url.replace("*", f"{info.folder}/{f}")
        total_size = info.pth_size if "consolidated" in f else 0
        tasks.append(DownloadTask(url=url, output_file=output_file, total_size=total_size, max_retries=3))

    # Initialize and run parallel downloader
    downloader = ParallelDownloader(max_concurrent_downloads=max_concurrent_downloads)
    asyncio.run(downloader.download_all(tasks))

    cprint(f"\nSuccessfully downloaded model to {output_dir}", color="green", file=sys.stderr)
    cprint(
        f"\nView MD5 checksum files at: {output_dir / 'checklist.chk'}",
        file=sys.stderr,
    )
    cprint(
        f"\n[Optionally] To run MD5 checksums, use the following command: llama model verify-download --model-id {model_id}",
        color="yellow",
        file=sys.stderr,
    )


class ModelEntry(BaseModel):
    model_id: str
    files: dict[str, str]

    model_config = ConfigDict(protected_namespaces=())


class Manifest(BaseModel):
    models: list[ModelEntry]
    expires_on: datetime


def _download_from_manifest(manifest_file: str, max_concurrent_downloads: int):
    from llama_stack.core.utils.model_utils import model_local_dir

    with open(manifest_file) as f:
        d = json.load(f)
        manifest = Manifest(**d)

    if datetime.now(UTC) > manifest.expires_on.astimezone(UTC):
        raise ValueError(f"Manifest URLs have expired on {manifest.expires_on}")

    console = Console()
    for entry in manifest.models:
        console.print(f"[blue]Downloading model {entry.model_id}...[/blue]")
        output_dir = Path(model_local_dir(entry.model_id))
        os.makedirs(output_dir, exist_ok=True)

        if any(output_dir.iterdir()):
            console.print(f"[yellow]Output directory {output_dir} is not empty.[/yellow]")

            while True:
                resp = input("Do you want to (C)ontinue download or (R)estart completely? (continue/restart): ")
                if resp.lower() in ["restart", "r"]:
                    shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    break
                elif resp.lower() in ["continue", "c"]:
                    console.print("[blue]Continuing download...[/blue]")
                    break
                else:
                    console.print("[red]Invalid response. Please try again.[/red]")

        # Create download tasks for all files in the manifest
        tasks = [
            DownloadTask(url=url, output_file=str(output_dir / fname), max_retries=3)
            for fname, url in entry.files.items()
        ]

        # Initialize and run parallel downloader
        downloader = ParallelDownloader(max_concurrent_downloads=max_concurrent_downloads)
        asyncio.run(downloader.download_all(tasks))


def run_download_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser):
    """Main download command handler"""
    try:
        if args.manifest_file:
            _download_from_manifest(args.manifest_file, args.max_parallel)
            return

        if args.model_id is None:
            parser.error("Please provide a model id")
            return

        # Handle comma-separated model IDs
        model_ids = [model_id.strip() for model_id in args.model_id.split(",")]

        from llama_stack.models.llama.sku_list import llama_meta_net_info, resolve_model

        from .model.safety_models import (
            prompt_guard_download_info_map,
            prompt_guard_model_sku_map,
        )

        prompt_guard_model_sku_map = prompt_guard_model_sku_map()
        prompt_guard_download_info_map = prompt_guard_download_info_map()

        for model_id in model_ids:
            if model_id in prompt_guard_model_sku_map.keys():
                model = prompt_guard_model_sku_map[model_id]
                info = prompt_guard_download_info_map[model_id]
            else:
                model = resolve_model(model_id)
                if model is None:
                    parser.error(f"Model {model_id} not found")
                    continue
                info = llama_meta_net_info(model)

            if args.source == "huggingface":
                _hf_download(model, args.hf_token, args.ignore_patterns, parser)
            else:
                meta_url = args.meta_url or input(
                    f"Please provide the signed URL for model {model_id} you received via email "
                    f"after visiting https://www.llama.com/llama-downloads/ "
                    f"(e.g., https://llama3-1.llamameta.net/*?Policy...): "
                )
                if "llamameta.net" not in meta_url:
                    parser.error("Invalid Meta URL provided")
                _meta_download(model, model_id, meta_url, info, args.max_parallel)

    except Exception as e:
        parser.error(f"Download failed: {str(e)}")
