#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
#
# CI can direct the generated artifacts into an alternate checkout by passing
# --repo-root, allowing the trusted copy of this script to run from a separate
# worktree.

import argparse
import concurrent.futures
import importlib
import subprocess
import sys
from collections.abc import Iterable
from functools import partial
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn

_DEFAULT_REPO_ROOT = Path(__file__).parent.parent
REPO_ROOT = _DEFAULT_REPO_ROOT


def set_repo_root(repo_root: Path) -> None:
    """Update the global repository root used by helper functions."""

    global REPO_ROOT
    REPO_ROOT = repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distribution docs and YAML artifacts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_DEFAULT_REPO_ROOT,
        help="Repository root where generated artifacts should be written.",
    )
    return parser.parse_args()


class ChangedPathTracker:
    """Track a list of paths we may have changed."""

    def __init__(self):
        self._changed_paths = []

    def add_paths(self, *paths):
        for path in paths:
            path = str(path)
            if path not in self._changed_paths:
                self._changed_paths.append(path)

    def changed_paths(self):
        return self._changed_paths


def find_distro_dirs(distro_dir: Path) -> Iterable[Path]:
    """Find immediate subdirectories in the distributions folder."""
    if not distro_dir.exists():
        raise FileNotFoundError(f"Distributions directory not found: {distro_dir}")

    return sorted(d for d in distro_dir.iterdir() if d.is_dir() and d.name != "__pycache__")


def process_distro(distro_dir: Path, progress, change_tracker: ChangedPathTracker) -> None:
    """Process a single distribution directory."""
    progress.print(f"Processing {distro_dir.name}")

    try:
        # Import the module directly
        module_name = f"llama_stack.distributions.{distro_dir.name}"
        module = importlib.import_module(module_name)

        # Get and save the distribution template
        if template_func := getattr(module, "get_distribution_template", None):
            distro = template_func()

            yaml_output_dir = REPO_ROOT / "src" / "llama_stack" / "distributions" / distro.name
            doc_output_dir = REPO_ROOT / "docs/docs/distributions" / f"{distro.distro_type}_distro"
            change_tracker.add_paths(yaml_output_dir, doc_output_dir)
            distro.save_distribution(
                yaml_output_dir=yaml_output_dir,
                doc_output_dir=doc_output_dir,
            )
        else:
            progress.print(f"[yellow]Warning: {distro_dir.name} has no get_distribution_template function")

    except Exception as e:
        progress.print(f"[red]Error processing {distro_dir.name}: {str(e)}")
        raise e


def check_for_changes(change_tracker: ChangedPathTracker) -> bool:
    """Check if there are any uncommitted changes."""
    has_changes = False
    for path in change_tracker.changed_paths():
        result = subprocess.run(
            ["git", "diff", "--exit-code", path],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Change detected in '{path}'.", file=sys.stderr)
            has_changes = True
    return has_changes


def pre_import_distros(distro_dirs: list[Path]) -> None:
    # Pre-import all distro modules to avoid deadlocks.
    for distro_dir in distro_dirs:
        module_name = f"llama_stack.distributions.{distro_dir.name}"
        importlib.import_module(module_name)


def main():
    args = parse_args()
    repo_root = args.repo_root
    if not repo_root.is_absolute():
        repo_root = (Path.cwd() / repo_root).resolve()

    set_repo_root(repo_root)

    distros_dir = REPO_ROOT / "src" / "llama_stack" / "distributions"
    change_tracker = ChangedPathTracker()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        distro_dirs = list(find_distro_dirs(distros_dir))
        task = progress.add_task("Processing distribution templates...", total=len(distro_dirs))

        pre_import_distros(distro_dirs)

        # Create a partial function with the progress bar
        process_func = partial(process_distro, progress=progress, change_tracker=change_tracker)

        # Process distributions in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks and wait for completion
            list(executor.map(process_func, distro_dirs))
            progress.update(task, advance=len(distro_dirs))

    if check_for_changes(change_tracker):
        print(
            "Distribution changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
