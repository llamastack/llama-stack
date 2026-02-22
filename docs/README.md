# Llama Stack Documentation

Here's a collection of comprehensive guides, examples, and resources for building AI applications with Llama Stack. For the complete documentation, visit our [Github page](https://llamastack.github.io/getting_started/quickstart).

## Render locally

From the repository root, use the build script:
```bash
./scripts/build-docs.sh --serve
```

Or from the `docs/` directory:
```bash
npm install
npm run gen-api-docs all
npm run build
npm run serve
```

You can open up the docs in your browser at http://localhost:3000

## API Documentation Generation

The API documentation is generated from OpenAPI specs using `docusaurus-plugin-openapi-docs`. The generation is optimized with hash-based caching to skip regeneration when specs haven't changed.

### npm Scripts

| Script | Description |
|--------|-------------|
| `npm run gen-api-docs` | Generate API docs with caching (skips if specs unchanged) |
| `npm run gen-api-docs:force` | Force regeneration of all API docs |
| `npm run gen-api-docs:parallel` | Generate API docs in parallel |
| `npm run gen-api-docs:stable` | Generate only stable API docs |
| `npm run gen-api-docs:experimental` | Generate only experimental API docs |
| `npm run gen-api-docs:deprecated` | Generate only deprecated API docs |
| `npm run gen-api-docs:raw` | Bypass caching and run docusaurus directly |

### Build Script Options

The `./scripts/build-docs.sh` script supports these options:

| Option | Description |
|--------|-------------|
| `--force` | Force regeneration of API docs |
| `--parallel` | Generate API docs in parallel |
| `--skip-api` | Skip API docs generation entirely |
| `--serve` | Start local server after building |

Environment variables `SKIP_API_DOCS=1` and `FORCE_API_DOCS=1` are also supported.

## File Import System

This documentation uses `remark-code-import` to import files directly from the repository, eliminating copy-paste maintenance. Files are automatically embedded during build time.

### Importing Code Files

To import Python code (or any code files) with syntax highlighting, use this syntax in `.mdx` files:

```markdown
```python file=./demo_script.py title="demo_script.py"
```
```

This automatically imports the file content and displays it as a formatted code block with Python syntax highlighting.

**Note:** Paths are relative to the current `.mdx` file location, not the repository root.

### Importing Markdown Files as Content

For importing and rendering markdown files (like CONTRIBUTING.md), use the raw-loader approach:

```jsx
import Contributing from '!!raw-loader!../../../CONTRIBUTING.md';
import ReactMarkdown from 'react-markdown';

<ReactMarkdown>{Contributing}</ReactMarkdown>
```

**Requirements:**
- Install dependencies: `npm install --save-dev raw-loader react-markdown`

**Path Resolution:**
- For `remark-code-import`: Paths are relative to the current `.mdx` file location
- For `raw-loader`: Paths are relative to the current `.mdx` file location
- Use `../` to navigate up directories as needed

## Content

Try out Llama Stack's capabilities through our detailed Jupyter notebooks:

* [Building AI Applications Notebook](./getting_started.ipynb) - A comprehensive guide to building production-ready AI applications using Llama Stack
* [Benchmark Evaluations Notebook](./notebooks/Llama_Stack_Benchmark_Evals.ipynb) - Detailed performance evaluations and benchmarking results
* [Zero-to-Hero Guide](./zero_to_hero_guide) - Step-by-step guide for getting started with Llama Stack
* [Migrating from Agent objects to Responses](./notebooks/responses-api-agent-migration.ipynb) - A guide to migrating from legacy Llama Stack Agent APIs to the OpenAI-compatible Responses API
