# remote::weaviate

## Description


[Weaviate](https://weaviate.io/) is a vector database provider for Llama Stack.
It allows you to store and query vectors directly within a Weaviate database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Weaviate supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Hybrid search
- Document storage
- Metadata filtering
- Multi-modal retrieval


## Usage

To use Weaviate in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

To install Weaviate see the [Weaviate quickstart documentation](https://weaviate.io/developers/weaviate/quickstart).

## Documentation
See [Weaviate's documentation](https://weaviate.io/developers/weaviate) for more details about Weaviate in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `weaviate_api_key` | `str \| None` | No |  | The API key for the Weaviate instance |
| `weaviate_cluster_url` | `str \| None` | No | localhost:8080 | The URL of the Weaviate cluster |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig, annotation=NoneType, required=False, default='sqlite', discriminator='type'` | No |  | Config for KV store backend (SQLite only for now) |

## Sample Configuration

```yaml
weaviate_api_key: null
weaviate_cluster_url: ${env.WEAVIATE_CLUSTER_URL:=localhost:8080}
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/weaviate_registry.db

```

