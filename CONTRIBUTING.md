# Contributing to Llama Stack
We want to make contributing to this project as easy and transparent as
possible.

## Set up your development environment

We use [uv](https://github.com/astral-sh/uv) to manage python dependencies and virtual environments.
You can install `uv` by following this [guide](https://docs.astral.sh/uv/getting-started/installation/).

You can install the dependencies by running:

```bash
cd llama-stack
uv venv --python 3.12
uv sync --group dev
uv pip install -e .
source .venv/bin/activate
```

```{note}
If you are making changes to Llama Stack, it is essential that you use Python 3.12 as shown above.
Llama Stack can work with Python 3.13 but the pre-commit hooks used to validate code changes only work with Python 3.12.
If you don't specify a Python version, `uv` will automatically select a Python version according to the `requires-python`
section of the `pyproject.toml`, which is fine for running Llama Stack but not for committing changes.
For more info, see the [uv docs around Python versions](https://docs.astral.sh/uv/concepts/python-versions/).
```

Note that you can create a dotenv file `.env` that includes necessary environment variables:
```
LLAMA_STACK_BASE_URL=http://localhost:8321
LLAMA_STACK_CLIENT_LOG=debug
LLAMA_STACK_PORT=8321
LLAMA_STACK_CONFIG=<provider-name>
TAVILY_SEARCH_API_KEY=
BRAVE_SEARCH_API_KEY=
```

And then use this dotenv file when running client SDK tests via the following:
```bash
uv run --env-file .env -- pytest -v tests/integration/inference/test_text_inference.py --text-model=meta-llama/Llama-3.1-8B-Instruct
```

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run linting and formatting checks on your code. You can install the pre-commit hooks by running:

```bash
uv pip install 'pre-commit>=4.4.0'
uv run pre-commit install
```

Note that the only version of pre-commit that works with the Llama Stack continuous integration is `4.3.0` so it is essential that you pull
that specific version as shown above.  Once you have run these commands, pre-commit hooks will run automatically before each commit.

Alternatively, if you don't want to install the pre-commit hooks (or if you want to check if your changes are ready before committing),
you can run the checks manually by running:

```bash
uv run pre-commit run --all-files -v
```

The `-v` (verbose) parameter is optional but often helpful for getting more information about any issues with that the pre-commit checks identify.

To run the expanded mypy configuration that CI enforces, use:

```bash
uv run pre-commit run mypy-full --hook-stage manual --all-files
```

or invoke mypy directly with all optional dependencies:

```bash
uv run --group dev --group type_checking mypy
```

```{caution}
Before pushing your changes, make sure that the pre-commit hooks have passed successfully.
```

## Discussions -> Issues -> Pull Requests

We actively welcome your pull requests. However, please read the following. This is heavily inspired by [Ghostty](https://github.com/ghostty-org/ghostty/blob/main/CONTRIBUTING.md).

If in doubt, please open a [discussion](https://github.com/llamastack/llama-stack/discussions); we can always convert that to an issue later.

### Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](http://facebook.com/whitehat/info) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

### Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

**I'd like to contribute!**

If you are new to the project, start by looking at the issues tagged with "good first issue". If you're interested
leave a comment on the issue and a triager will assign it to you.

Please avoid picking up too many issues at once. This helps you stay focused and ensures that others in the community also have opportunities to contribute.

- Try to work on only 1–2 issues at a time, especially if you’re still getting familiar with the codebase.
- Before taking an issue, check if it’s already assigned or being actively discussed.
- If you’re blocked or can’t continue with an issue, feel free to unassign yourself or leave a comment so others can step in.

**I have a bug!**

1. Search the issue tracker and discussions for similar issues.
2. If you don't have steps to reproduce, open a discussion.
3. If you have steps to reproduce, open an issue.

**I have an idea for a feature!**

1. Open a discussion.

**I've implemented a feature!**

1. If there is an issue for the feature, open a pull request.
2. If there is no issue, open a discussion and link to your branch.

**I have a question!**

1. Open a discussion or use [Discord](https://discord.gg/llama-stack).


**Opening a Pull Request**

1. Fork the repo and create your branch from `main`.
2. If you've changed APIs, update the documentation.
3. Ensure the test suite passes.
4. Make sure your code lints using `pre-commit`.
5. If you haven't already, complete the Contributor License Agreement ("CLA").
6. Ensure your pull request follows the [conventional commits format](https://www.conventionalcommits.org/en/v1.0.0/).
7. Ensure your pull request follows the [coding style](#coding-style).


Please keep pull requests (PRs) small and focused. If you have a large set of changes, consider splitting them into logically grouped, smaller PRs to facilitate review and testing.

```{tip}
As a general guideline:
- Experienced contributors should try to keep no more than 5 open PRs at a time.
- New contributors are encouraged to have only one open PR at a time until they’re familiar with the codebase and process.
```

## Repository guidelines

### Coding Style

* Comments should provide meaningful insights into the code. Avoid filler comments that simply
  describe the next step, as they create unnecessary clutter, same goes for docstrings.
* Prefer comments to clarify surprising behavior and/or relationships between parts of the code
  rather than explain what the next line of code does.
* Catching exceptions, prefer using a specific exception type rather than a broad catch-all like
  `Exception`.
* Error messages should be prefixed with "Failed to ..."
* 4 spaces for indentation rather than tab
* When using `# noqa` to suppress a style or linter warning, include a comment explaining the
  justification for bypassing the check.
* When using `# type: ignore` to suppress a mypy warning, include a comment explaining the
  justification for bypassing the check.
* Don't use unicode characters in the codebase. ASCII-only is preferred for compatibility or
  readability reasons.
* Providers configuration class should be Pydantic Field class. It should have a `description` field
  that describes the configuration. These descriptions will be used to generate the provider
  documentation.
* When possible, use keyword arguments only when calling functions.
* Llama Stack utilizes [custom Exception classes](llama_stack/apis/common/errors.py) for certain Resources that should be used where applicable.

### License
By contributing to Llama, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

## Common Tasks

Some tips about common tasks you work on while contributing to Llama Stack:

### Installing dependencies of distributions

When installing dependencies for a distribution, you can use `llama stack list-deps` to view and install the required packages.

Example:
```bash
cd work/
git clone https://github.com/llamastack/llama-stack.git
git clone https://github.com/llamastack/llama-stack-client-python.git
cd llama-stack

# Show dependencies for a distribution
llama stack list-deps <distro-name>

# Install dependencies
llama stack list-deps <distro-name> | xargs -L1 uv pip install
```

### Updating distribution configurations

If you have made changes to a provider's configuration in any form (introducing a new config key, or
changing models, etc.), you should run `./scripts/distro_codegen.py` to re-generate various YAML
files as well as the documentation. You should not change `docs/source/.../distributions/` files
manually as they are auto-generated.

### Updating the provider documentation

If you have made changes to a provider's configuration, you should run `./scripts/provider_codegen.py`
to re-generate the documentation. You should not change `docs/source/.../providers/` files manually
as they are auto-generated.
Note that the provider "description" field will be used to generate the provider documentation.

### Building the Documentation

If you are making changes to the documentation at [https://llamastack.github.io/](https://llamastack.github.io/), you can use the following command to build the documentation and preview your changes.

```bash
# This rebuilds the documentation pages and the OpenAPI spec.
cd docs/
npm install
npm run gen-api-docs all
npm run build

# This will start a local server (usually at http://127.0.0.1:3000).
npm run serve
```

### Update API Documentation

If you modify or add new API endpoints, update the API documentation accordingly. You can do this by running the following command:

```bash
uv run ./scripts/run_openapi_generator.sh
```

The generated API schema will be available in `docs/static/`. Make sure to review the changes before committing.
