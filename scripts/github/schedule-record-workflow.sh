#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Script to easily trigger the integration test recording workflow
# Usage: ./scripts/github/schedule-record-workflow.sh [options]

set -euo pipefail

# Default values
BRANCH=""
PR_HEAD_REPO=""
TEST_SUBDIRS=""
TEST_SETUP="ollama"
TEST_SUITE="base"
TEST_PATTERN=""

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Trigger the integration test recording workflow remotely. This way you do not need to have Ollama running locally.

OPTIONS:
    -b, --branch BRANCH         Branch to run the workflow on (defaults to current branch)
    -t, --suite SUITE           Test suite to use: base, responses, vision, etc. (default: base)
    -p, --setup SETUP           Test setup to use: vllm, ollama, gpt, etc. (default: ollama)
    -s, --subdirs DIRS          Comma-separated list of test subdirectories to run (overrides suite)
    -k, --pattern PATTERN       Regex pattern to pass to pytest -k
    -h, --help                  Show this help message

EXAMPLES:
    # Record tests for current branch with agents subdirectory
    $0 --subdirs "agents"

    # Record tests for specific branch with vision tests
    $0 -b my-feature-branch --suite vision

    # Record multiple test subdirectories with specific setup
    $0 --subdirs "agents,inference" --setup vllm

    # Record tests matching a specific pattern
    $0 --subdirs "inference" --pattern "test_streaming"

EOF
}

# PREREQUISITES:
#     - GitHub CLI (gh) must be installed and authenticated
#     - jq must be installed for JSON parsing
#     - You must be in a git repository that is a fork or clone of llamastack/llama-stack
#     - The branch must exist on the remote repository where you want to run the workflow
#     - You must specify test subdirectories to run with -s/--test-subdirs

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -s|--subdirs)
            TEST_SUBDIRS="$2"
            shift 2
            ;;
        -p|--setup)
            TEST_SETUP="$2"
            shift 2
            ;;
        -t|--suite)
            TEST_SUITE="$2"
            shift 2
            ;;
        -k|--pattern)
            TEST_PATTERN="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$TEST_SUBDIRS" && -z "$TEST_SUITE" ]]; then
    echo "Error: --subdirs or --suite is required"
    echo "Please specify which test subdirectories to run or test suite to use, e.g.:"
    echo "  $0 --subdirs \"agents,inference\""
    echo "  $0 --suite vision"
    echo ""
    exit 1
fi

# Validate test setup (optional - setups are validated by the workflow itself)
# Common setups: ollama, vllm, gpt, etc.

# Check if required tools are installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed. Please install it from https://cli.github.com/"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "Error: GitHub CLI is not authenticated. Please run 'gh auth login'"
    exit 1
fi

# If no branch specified, use current branch
if [[ -z "$BRANCH" ]]; then
    BRANCH=$(git branch --show-current)
    echo "No branch specified, using current branch: $BRANCH"

    # Optionally look for associated PR for context (not required)
    echo "Looking for associated PR..."

    # Search for PRs in the main repo that might match this branch
    # This searches llamastack/llama-stack for any PR with this head branch name
    if PR_INFO=$(gh pr list --repo llamastack/llama-stack --head "$BRANCH" --json number,headRefName,headRepository,headRepositoryOwner,url,state --limit 1 2>/dev/null) && [[ "$PR_INFO" != "[]" ]]; then
        # Parse PR info using jq
        PR_NUMBER=$(echo "$PR_INFO" | jq -r '.[0].number')
        PR_HEAD_REPO=$(echo "$PR_INFO" | jq -r '.[0].headRepositoryOwner.login // "llamastack"')
        PR_URL=$(echo "$PR_INFO" | jq -r '.[0].url')
        PR_STATE=$(echo "$PR_INFO" | jq -r '.[0].state')

        if [[ -n "$PR_NUMBER" && -n "$PR_HEAD_REPO" ]]; then
            echo "✅ Found associated PR #$PR_NUMBER ($PR_STATE)"
            echo "   URL: $PR_URL"
            echo "   Head repository: $PR_HEAD_REPO/llama-stack"

            # Check PR state and block if merged
            if [[ "$PR_STATE" == "CLOSED" ]]; then
                echo "ℹ️  Note: This PR is closed, but workflow can still run to update recordings."
            elif [[ "$PR_STATE" == "MERGED" ]]; then
                echo "❌ Error: This PR is already merged."
                echo "   Cannot record tests for a merged PR since changes can't be committed back."
                echo "   Create a new branch/PR if you need to record new tests."
                exit 1
            fi
        fi
    else
        echo "ℹ️  No associated PR found for branch '$BRANCH'"
        echo "That's fine - the workflow just needs a pushed branch to run."
    fi
    echo ""
fi

# Determine the target repository for workflow dispatch based on where the branch actually exists
# We need to find which remote has the branch we want to run the workflow on

echo "Determining target repository for workflow..."

# Check if we have PR info with head repository
if [[ -n "$PR_HEAD_REPO" ]]; then
    # Use the repository from the PR head
    TARGET_REPO="$PR_HEAD_REPO/llama-stack"
    echo "📍 Using PR head repository: $TARGET_REPO"

    if [[ "$PR_HEAD_REPO" == "llamastack" ]]; then
        REPO_CONTEXT=""
    else
        REPO_CONTEXT="--repo $TARGET_REPO"
    fi
else
    # Fallback: find which remote has the branch
    BRANCH_REMOTE=""
    for remote in $(git remote); do
        if git ls-remote --heads "$remote" "$BRANCH" | grep -q "$BRANCH"; then
            REMOTE_URL=$(git remote get-url "$remote")
            if [[ "$REMOTE_URL" == *"/llama-stack"* ]]; then
                REPO_OWNER=$(echo "$REMOTE_URL" | sed -n 's/.*[:/]\([^/]*\)\/llama-stack.*/\1/p')
                echo "📍 Found branch '$BRANCH' on remote '$remote' ($REPO_OWNER/llama-stack)"
                TARGET_REPO="$REPO_OWNER/llama-stack"
                BRANCH_REMOTE="$remote"
                break
            fi
        fi
    done

    if [[ -z "$BRANCH_REMOTE" ]]; then
        echo "Error: Could not find branch '$BRANCH' on any llama-stack remote"
        echo ""
        echo "This could mean:"
        echo "   - The branch doesn't exist on any remote yet (push it first)"
        echo "   - The branch name is misspelled"
        echo "   - No llama-stack remotes are configured"
        echo ""
        echo "Available remotes:"
        git remote -v
        echo ""
        echo "To push your branch: git push <remote> $BRANCH"
        echo "Common remotes to try: origin, upstream, your-username"
        exit 1
    fi

    if [[ "$TARGET_REPO" == "llamastack/llama-stack" ]]; then
        REPO_CONTEXT=""
    else
        REPO_CONTEXT="--repo $TARGET_REPO"
    fi
fi

echo "   Workflow will run on: $TARGET_REPO"

# Verify the target repository has the workflow file
echo "Verifying workflow exists on target repository..."
if ! gh api "repos/$TARGET_REPO/contents/.github/workflows/record-integration-tests.yml" &>/dev/null; then
    echo "Error: The recording workflow does not exist on $TARGET_REPO"
    echo "This could mean:"
    echo "   - The fork doesn't have the latest workflow file"
    echo "   - The workflow file was renamed or moved"
    echo ""
    if [[ "$TARGET_REPO" != "llamastack/llama-stack" ]]; then
        echo "Try syncing your fork with upstream:"
        echo "   git fetch upstream"
        echo "   git checkout main"
        echo "   git merge upstream/main"
        echo "   git push origin main"
    fi
    exit 1
fi

# Build the workflow dispatch command
echo "Triggering integration test recording workflow..."
echo "Branch: $BRANCH"
echo "Test setup: $TEST_SETUP"
echo "Test subdirs: $TEST_SUBDIRS"
echo "Test suite: $TEST_SUITE"
echo "Test pattern: ${TEST_PATTERN:-"(none)"}"
echo ""

# Prepare inputs for gh workflow run
INPUTS=
if [[ -n "$TEST_SUBDIRS" ]]; then
    INPUTS="$INPUTS -f subdirs='$TEST_SUBDIRS'"
fi
if [[ -n "$TEST_SETUP" ]]; then
    INPUTS="$INPUTS -f test-setup='$TEST_SETUP'"
fi
if [[ -n "$TEST_SUITE" ]]; then
    INPUTS="$INPUTS -f suite='$TEST_SUITE'"
fi
if [[ -n "$TEST_PATTERN" ]]; then
    INPUTS="$INPUTS -f pattern='$TEST_PATTERN'"
fi

# Run the workflow
WORKFLOW_CMD="gh workflow run record-integration-tests.yml --ref $BRANCH $REPO_CONTEXT $INPUTS"
echo "Running: $WORKFLOW_CMD"
echo ""

if eval "$WORKFLOW_CMD"; then
    echo "✅ Workflow triggered successfully!"
    echo ""
    echo "You can monitor the workflow run at:"
    echo "https://github.com/$TARGET_REPO/actions/workflows/record-integration-tests.yml"
    echo ""
    if [[ -n "$REPO_CONTEXT" ]]; then
        echo "Or use: gh run list --workflow=record-integration-tests.yml $REPO_CONTEXT"
        echo "And then: gh run watch <RUN_ID> $REPO_CONTEXT"
    else
        echo "Or use: gh run list --workflow=record-integration-tests.yml"
        echo "And then: gh run watch <RUN_ID>"
    fi
else
    echo "❌ Failed to trigger workflow"
    exit 1
fi
