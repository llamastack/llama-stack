# inline::faiss

## Description


[Faiss](https://github.com/facebookresearch/faiss) is an inline vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Lightweight and easy to use
- Fully integrated with Llama Stack
- GPU support
- **Vector search** - FAISS supports pure vector similarity search using embeddings

## Search Modes

**Supported:**
- **Vector Search** (`mode="vector"`): Performs vector similarity search using embeddings

**Not Supported:**
- **Keyword Search** (`mode="keyword"`): Not supported by FAISS
- **Hybrid Search** (`mode="hybrid"`): Not supported by FAISS

> **Note**: FAISS is designed as a pure vector similarity search library. See the [FAISS GitHub repository](https://github.com/facebookresearch/faiss) for more details about FAISS's core functionality.

## Usage

To use Faiss in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use Faiss.
3. Start storing and querying vectors.

## Installation

You can install Faiss using pip:

```bash
pip install faiss-cpu
```
## Documentation
See [Faiss' documentation](https://faiss.ai/) or the [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) for
more details about Faiss in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig` | No | sqlite |  |

## Sample Configuration

```yaml
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/faiss_store.db

```

