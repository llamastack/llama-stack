# Agent Guidelines for Llama Stack

This file provides guidance for AI coding agents working on this codebase.
It complements [CONTRIBUTING.md](CONTRIBUTING.md) with agent-specific instructions.
Human contributors should follow the conventions in CONTRIBUTING.md.

## Project Overview

Llama Stack is an API server implementing the OpenAI Responses API, Chat Completions,
Embeddings, and supporting APIs (files, vector stores, batches, eval, safety). It supports
multiple inference backends (OpenAI, Azure, Bedrock, vLLM, Ollama, WatsonX, etc.) through
a provider architecture.

## Repository Layout

```text
src/llama_stack/              # Server implementation
  core/                       # Request routing, server, storage
  providers/
    inline/                   # Built-in providers (responses, eval, vector_io, etc.)
    remote/                   # Remote provider adapters (OpenAI, Azure, vLLM, etc.)
    registry/                 # Provider registration specs
    utils/                    # Shared provider utilities (OpenAI mixin, MCP, etc.)
  distributions/              # Distribution configs (starter, ci-tests, etc.)
src/llama_stack_api/          # API definitions, Pydantic models, FastAPI routes
tests/
  unit/                       # Unit tests
  integration/                # Integration tests with recording/replay system
```

## Python & Tooling

- **Python 3.12** is required. Pre-commit hooks only work with 3.12.
- Use `uv` for all dependency management and script execution.
- Run scripts with `uv run`, never bare `python3` or `python`.
- Use standard library modules when possible.
- All function signatures must have type hints. Prefer Pydantic models for validation.
- Code must pass `mypy`. Check the exclude list in `pyproject.toml` for known exceptions.
- Use `def _function_name` for private functions.
- Prefer explicit top-level imports over inline imports.
- Do not use exceptions as control flow.

## Code Style

- Comments must add value. Do not write filler comments that describe the next line.
- Add comments only to clarify surprising behavior that is not obvious from the code.
- Good variable naming and clear code organization matters more than comments.
- Do NOT remove existing comments unless they are factually wrong.
- Error messages must be prefixed with "Failed to ...".
- Use structured logging via `from llama_stack.log import get_logger`. Always use
  key-value style: `logger.info("Processing request", model=model_id, provider=provider)`
  instead of f-strings or %-style formatting. The pre-commit hook
  `Block f-string logging` enforces this.
- The pre-commit hook `Ensure 'llama_stack.log' usage for logging` enforces that all
  logging uses the project's logger, not the standard library directly.

## Git Conventions

- Always use `--signoff` (`-s`) when creating commits.
- Do not amend commits and force push during PR review. Use new commits instead.
- Use `git merge main` to update branches, not `git rebase`.
- Commit messages must use [conventional commits](https://www.conventionalcommits.org/)
  format and full sentences, not bullet points.
- Merge `upstream/main` before pushing a branch to avoid CI failures from stale code.

## Testing

- Unit tests: `./scripts/unit-tests.sh` or `uv run pytest tests/unit/ -x --tb=short`
- Integration tests: `./scripts/integration-tests.sh` with recording/replay system.
  Recordings are JSON files in `tests/integration/*/recordings/`. When modifying code
  that changes request bodies sent to providers, recordings may need to be re-recorded.
- Run pre-commit checks: `uv run pre-commit run --all-files`

## Provider Architecture

Each provider implements a protocol (e.g., `Inference`, `Responses`, `VectorIO`) and is
registered in `src/llama_stack/providers/registry/`. Provider specs include:

- `provider_type`: e.g., `remote::openai`, `inline::builtin`
- `module`: Python module path
- `config_class`: Pydantic config class path
- `api_dependencies`: APIs this provider depends on
- `deprecation_warning`: For deprecated providers (triggers runtime warning)
- `toolgroup_id`: For tool_runtime providers that auto-register tool groups

Configuration classes must use Pydantic `Field` with `description` parameters —
these generate the provider documentation automatically.

## Distribution Configs

Distribution YAML files in `src/llama_stack/distributions/` are partially auto-generated.
After changing provider configs, run:

```bash
uv run ./scripts/distro_codegen.py
uv run ./scripts/provider_codegen.py
```

Do not edit generated files in `docs/docs/providers/` manually.

## API Changes

When modifying or extending APIs:

1. Update models in `src/llama_stack_api/`
2. Regenerate OpenAPI specs: `uv run ./scripts/run_openapi_generator.sh`
3. Check for breaking changes — the pre-commit hook `Check API spec for breaking changes`
   enforces backward compatibility.
4. Include a test plan with a testing script and execution output in your PR description.

## Common Patterns

### Adding a new parameter to an existing API

1. Add the field to the Pydantic model in `src/llama_stack_api/`
2. Thread it through the provider protocol and implementation
3. Update affected distribution configs if needed
4. Regenerate specs and docs
5. Add unit test cases covering the new parameter

### Adding a deprecated alias for a renamed provider

Use the existing `deprecation_warning` field on `InlineProviderSpec` or `RemoteProviderSpec`.
Search for existing examples: `grep -r "deprecation_warning" src/llama_stack/providers/registry/`

### Before adding any new pattern

Search the codebase for existing examples of the same pattern first.
Use `grep` to find how deprecation, validation, configuration, or aliasing is already
handled elsewhere.

## Keeping Documentation Up to Date

When making code changes, check whether the following documentation needs updating:

- `ARCHITECTURE.md` — system overview, request flow, provider architecture, API layer,
  storage, configuration, and test recording system
- Module-level `README.md` files in key directories:
  - `src/llama_stack/README.md`, `src/llama_stack/core/README.md`
  - `src/llama_stack/core/server/README.md`, `src/llama_stack/core/storage/README.md`
  - `src/llama_stack/core/routing_tables/README.md`
  - `src/llama_stack/providers/README.md`, `src/llama_stack/providers/inline/README.md`
  - `src/llama_stack/providers/remote/README.md`, `src/llama_stack/providers/registry/README.md`
  - `src/llama_stack/providers/utils/README.md`, `src/llama_stack/providers/utils/inference/README.md`
  - `src/llama_stack/providers/inline/agents/README.md`
  - `src/llama_stack/providers/inline/tool_runtime/README.md`
  - `src/llama_stack/providers/remote/inference/README.md`
  - `src/llama_stack/distributions/README.md`
  - `scripts/README.md`
- `tests/README.md`, `tests/unit/README.md`, `tests/integration/README.md`

These files help LLMs and new contributors navigate the codebase. If your change
adds, removes, or renames modules, providers, APIs, or storage backends, update
the relevant documentation to stay in sync.
