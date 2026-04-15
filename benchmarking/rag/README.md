# RAG Benchmark Suite

Evaluates RAG quality through Llama Stack's OpenAI-compatible API surface
(Files → Vector Stores → Responses with `file_search`). The same code runs
against both **OpenAI SaaS** and **Llama Stack** — swap `--base-url` to
switch backends.

## Benchmarks

| Benchmark | Type | Primary Metric | Description |
|---|---|---|---|
| **BEIR** | Retrieval-only | nDCG@10 | Standard IR benchmarks (nfcorpus, scifact, arguana, fiqa, trec-covid) |
| **MultiHOP RAG** | End-to-end RAG | EM / F1 | Multi-hop reasoning over news articles |
| **QReCC** | Conversational RAG | EM / F1 | Multi-turn conversational QA with scoped corpus per conversation |
| **Doc2Dial** | Document-grounded dialogue | EM / F1 | Goal-oriented dialogues grounded in documents |

## Prerequisites

- Python 3.11+
- An OpenAI API key (for OpenAI SaaS runs, or for Llama Stack's `remote::openai` inference)
- For Llama Stack runs: a running Milvus instance and a Llama Stack server

## Quick Start

```bash
cd benchmarking/rag

# Create a virtual environment and install dependencies
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Copy and edit the environment file
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## Running Against OpenAI

```bash
python run_benchmark.py --benchmark beir --base-url https://api.openai.com/v1
python run_benchmark.py --benchmark multihop --base-url https://api.openai.com/v1
python run_benchmark.py --benchmark qrecc --base-url https://api.openai.com/v1
python run_benchmark.py --benchmark doc2dial --base-url https://api.openai.com/v1
```

## Running Against Llama Stack

First, start Llama Stack with the included config:

```bash
# Start Milvus (if not already running)
# Start Llama Stack
bash start_stack.sh
```

Then run benchmarks:

```bash
python run_benchmark.py --benchmark beir --base-url http://localhost:8321/v1 --search-mode hybrid
python run_benchmark.py --benchmark multihop --base-url http://localhost:8321/v1 --search-mode hybrid
```

## Running All Benchmarks

```bash
bash run_all.sh
```

This runs all four benchmarks against both OpenAI SaaS and Llama Stack
(vector and hybrid search modes).

## Comparing Results

After running benchmarks against multiple backends:

```bash
python compare_results.py              # Table output
python compare_results.py --format csv # CSV output
python compare_results.py --format json --output results.json
```

## CLI Options

```text
python run_benchmark.py --help

Options:
  --benchmark       beir | multihop | qrecc | doc2dial (required)
  --dataset         BEIR dataset name (nfcorpus, scifact, arguana, fiqa, trec-covid)
  --base-url        API base URL (default: https://api.openai.com/v1)
  --search-mode     vector | hybrid | keyword (Llama Stack only)
  --model           Model for end-to-end RAG (default: gpt-4.1)
  --max-queries     Limit number of queries (useful for testing)
  --max-conversations  Limit conversations for QReCC/Doc2Dial
  --resume          Resume from checkpoint
  --output-dir      Custom output directory
  --data-dir        Custom data directory
  --verbose         Enable debug logging
```

## Directory Layout

```text
benchmarking/rag/
├── lib/                  # Core library
│   ├── client.py         # OpenAI client factory (configurable base_url)
│   ├── ingest.py         # Corpus upload: Files API → Vector Stores API
│   ├── search.py         # Retrieval-only eval: Vector Stores Search API
│   ├── query.py          # End-to-end RAG: Responses API with file_search
│   ├── metrics.py        # pytrec_eval, HF evaluate, rouge-score
│   └── utils.py          # Checkpointing, batching, ID mapping
├── benchmarks/           # Benchmark adapters
│   ├── base.py           # Abstract BenchmarkRunner
│   ├── beir_bench.py     # BEIR (retrieval-only, nDCG@10)
│   ├── multihop_bench.py # MultiHOP RAG (end-to-end, EM/F1)
│   ├── qrecc_bench.py    # QReCC (conversational, scoped corpus)
│   └── doc2dial_bench.py # Doc2Dial (document-grounded dialogue)
├── run_benchmark.py      # CLI entry point
├── run_all.sh            # Run all benchmarks
├── compare_results.py    # Generate comparison tables
├── config.yaml           # Llama Stack server config
├── start_stack.sh        # Launch Llama Stack
├── requirements.txt      # Python dependencies
└── .env.example          # Environment variables template
```

## How It Works

1. **Ingestion**: Documents are uploaded via the Files API, then attached to a
   Vector Store via the Vector Stores API. Uploads are batched and checkpointed
   for resumability.

2. **Retrieval** (BEIR): Queries are run through the Vector Stores Search API.
   Results are mapped back to corpus doc IDs and evaluated with pytrec_eval.

3. **End-to-end RAG** (MultiHOP, QReCC, Doc2Dial): Queries are sent to the
   Responses API with the `file_search` tool. Answer quality is measured with
   Exact Match, token-level F1 (HuggingFace SQuAD metric), and ROUGE-L.

4. **Conversational RAG** (QReCC, Doc2Dial): Multi-turn conversations are
   threaded using `previous_response_id` to maintain context across turns.
