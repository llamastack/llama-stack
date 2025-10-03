#!/usr/bin/env python3
"""
Utility script to re-normalize existing recording files.

This script reads all recording JSON files and applies the normalization
to make IDs deterministic and timestamps constant. This reduces noise in
git diffs when recordings are re-recorded.

Usage:
    python scripts/normalize_recordings.py [--dry-run]
"""
import argparse
import json
from pathlib import Path


def normalize_tool_call_ids(obj, request_hash: str, counter: dict) -> None:
    """Recursively normalize tool call IDs in an object structure."""
    if isinstance(obj, dict):
        # Normalize tool_calls array
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
            for tool_call in obj["tool_calls"]:
                if isinstance(tool_call, dict) and "id" in tool_call:
                    # Generate deterministic tool call ID
                    tool_call["id"] = f"toolcall-{request_hash[:8]}-{counter['count']}"
                    counter["count"] += 1
                # Recurse into nested structures
                normalize_tool_call_ids(tool_call, request_hash, counter)

        # Normalize tool_call_id field (used in tool response messages)
        if "tool_call_id" in obj:
            # We need to map this to the same ID that was used in tool_calls
            # For post-facto cleanup, we'll use a simple approach: extract index if possible
            # Otherwise just normalize with counter
            obj["tool_call_id"] = f"toolcall-{request_hash[:8]}-{counter['tool_response_count']}"
            counter["tool_response_count"] += 1

        # Recurse into all dict values
        for key, value in obj.items():
            if key not in ("tool_calls", "tool_call_id"):  # Already handled above
                normalize_tool_call_ids(value, request_hash, counter)

    elif isinstance(obj, list):
        # Recurse into list items
        for item in obj:
            normalize_tool_call_ids(item, request_hash, counter)


def normalize_response_data(data: dict, request_hash: str) -> dict:
    """Normalize fields that change between recordings but don't affect functionality."""
    # Only normalize ID for completion/chat responses, not for model objects
    # Model objects have "object": "model" and the ID is the actual model identifier
    if "id" in data and data.get("object") != "model":
        data["id"] = f"rec-{request_hash[:12]}"

    # Normalize timestamp to epoch (0) (for OpenAI-style responses)
    # But not for model objects where created timestamp might be meaningful
    if "created" in data and data.get("object") != "model":
        data["created"] = 0

    # Normalize Ollama-specific timestamp fields
    if "created_at" in data:
        data["created_at"] = "1970-01-01T00:00:00.000000Z"

    # Normalize Ollama-specific duration fields (these vary based on system load)
    if "total_duration" in data and data["total_duration"] is not None:
        data["total_duration"] = 0
    if "load_duration" in data and data["load_duration"] is not None:
        data["load_duration"] = 0
    if "prompt_eval_duration" in data and data["prompt_eval_duration"] is not None:
        data["prompt_eval_duration"] = 0
    if "eval_duration" in data and data["eval_duration"] is not None:
        data["eval_duration"] = 0

    # Normalize tool call IDs
    counter = {"count": 0, "tool_response_count": 0}
    normalize_tool_call_ids(data, request_hash, counter)

    return data


def normalize_recording_file(file_path: Path, dry_run: bool = False) -> bool:
    """Normalize a single recording file. Returns True if file was modified."""
    with open(file_path) as f:
        recording = json.load(f)

    # Extract request hash from filename (first 12 chars)
    request_hash = file_path.stem.split("-")[-1] if "-" in file_path.stem else file_path.stem

    modified = False
    old_recording = json.dumps(recording, sort_keys=True)

    # NOTE: We do NOT normalize request body here because that would change the request hash
    # and break recording lookups. The recorder will normalize tool_call_ids in future recordings.

    # Normalize response body
    if "response" in recording and "body" in recording["response"]:
        body = recording["response"]["body"]

        if isinstance(body, list):
            # Handle streaming responses (list of chunks)
            for chunk in body:
                if isinstance(chunk, dict) and "__data__" in chunk:
                    normalize_response_data(chunk["__data__"], request_hash)
        elif isinstance(body, dict) and "__data__" in body:
            # Handle single response
            normalize_response_data(body["__data__"], request_hash)

    # Check if anything changed
    new_recording = json.dumps(recording, sort_keys=True)
    modified = old_recording != new_recording

    if modified and not dry_run:
        with open(file_path, "w") as f:
            json.dump(recording, f, indent=2)
            f.write("\n")

    return modified


def main():
    parser = argparse.ArgumentParser(description="Normalize recording files to reduce git diff noise")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    recordings_dir = Path(__file__).parent.parent / "tests/integration/recordings/responses"

    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return 1

    modified_count = 0
    total_count = 0

    for file_path in sorted(recordings_dir.glob("*.json")):
        total_count += 1
        was_modified = normalize_recording_file(file_path, dry_run=args.dry_run)

        if was_modified:
            modified_count += 1
            status = "[DRY RUN] Would normalize" if args.dry_run else "Normalized"
            print(f"{status}: {file_path.name}")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary: {modified_count}/{total_count} files modified")
    return 0


if __name__ == "__main__":
    exit(main())
