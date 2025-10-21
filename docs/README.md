# Llama Stack Documentation

Here's a collection of comprehensive guides, examples, and resources for building AI applications with Llama Stack. For the complete documentation, visit our [Github page](https://llamastack.github.io/getting_started/quickstart).

## Render locally

From the llama-stack `docs/` directory, run the following commands to render the docs locally:
```bash
npm install
npm run gen-api-docs all
npm run build
npm run serve
```
You can open up the docs in your browser at http://localhost:3000

## File Import System

This documentation uses a custom component to import files directly from the repository, eliminating copy-paste maintenance:

```jsx
import CodeFromFile from '@site/src/components/CodeFromFile';

<CodeFromFile src="path/to/file.py" />
<CodeFromFile src="README.md" startLine={1} endLine={20} />
```

Files are automatically synced from the repo root when building. See the `CodeFromFile` component for syntax highlighting, line ranges, and multi-language support.

## Content

Try out Llama Stack's capabilities through our detailed Jupyter notebooks:

* [Building AI Applications Notebook](./getting_started.ipynb) - A comprehensive guide to building production-ready AI applications using Llama Stack
* [Benchmark Evaluations Notebook](./notebooks/Llama_Stack_Benchmark_Evals.ipynb) - Detailed performance evaluations and benchmarking results
* [Zero-to-Hero Guide](./zero_to_hero_guide) - Step-by-step guide for getting started with Llama Stack
