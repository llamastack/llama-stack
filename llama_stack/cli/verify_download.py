# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from llama_stack.cli.subcommand import Subcommand


@dataclass
class VerificationResult:
    filename: str
    expected_hash: str
    actual_hash: str | None
    exists: bool
    matches: bool


class VerifyDownload(Subcommand):
    """Llama cli for verifying downloaded model files"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "verify-download",
            prog="llama verify-download",
            description="Verify integrity of downloaded model files",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        setup_verify_download_parser(self.parser)


def setup_verify_download_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID to verify (only for models downloaded from Meta)",
    )
    parser.set_defaults(func=partial(run_verify_cmd, parser=parser))


def calculate_md5(filepath: Path, chunk_size: int = 8192) -> str:
    # NOTE: MD5 is used here only for download integrity verification,
    # not for security purposes
    # TODO: switch to SHA256
    md5_hash = hashlib.md5(usedforsecurity=False)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def load_checksums(checklist_path: Path) -> dict[str, str]:
    checksums = {}
    with open(checklist_path) as f:
        for line in f:
            if line.strip():
                md5sum, filepath = line.strip().split("  ", 1)
                # Remove leading './' if present
                filepath = filepath.lstrip("./")
                checksums[filepath] = md5sum
    return checksums


def verify_files(model_dir: Path, checksums: dict[str, str], console: Console) -> list[VerificationResult]:
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for filepath, expected_hash in checksums.items():
            full_path = model_dir / filepath
            task_id = progress.add_task(f"Verifying {filepath}...", total=None)

            exists = full_path.exists()
            actual_hash = None
            matches = False

            if exists:
                actual_hash = calculate_md5(full_path)
                matches = actual_hash == expected_hash

            results.append(
                VerificationResult(
                    filename=filepath,
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                    exists=exists,
                    matches=matches,
                )
            )

            progress.remove_task(task_id)

    return results


def run_verify_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser):
    from llama_stack.core.utils.model_utils import model_local_dir

    console = Console()
    model_dir = Path(model_local_dir(args.model_id))
    checklist_path = model_dir / "checklist.chk"

    if not model_dir.exists():
        parser.error(f"Model directory not found: {model_dir}")

    if not checklist_path.exists():
        parser.error(f"Checklist file not found: {checklist_path}")

    checksums = load_checksums(checklist_path)
    results = verify_files(model_dir, checksums, console)

    # Print results
    console.print("\nVerification Results:")

    all_good = True
    for result in results:
        if not result.exists:
            console.print(f"[red]❌ {result.filename}: File not found[/red]")
            all_good = False
        elif not result.matches:
            console.print(
                f"[red]❌ {result.filename}: Hash mismatch[/red]\n"
                f"   Expected: {result.expected_hash}\n"
                f"   Got:      {result.actual_hash}"
            )
            all_good = False
        else:
            console.print(f"[green]✓ {result.filename}: Verified[/green]")

    if all_good:
        console.print("\n[green]All files verified successfully![/green]")
