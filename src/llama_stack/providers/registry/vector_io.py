# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    RemoteProviderSpec,
)

# Common dependencies for all vector IO providers that support document processing
DEFAULT_VECTOR_IO_DEPS = ["chardet", "pypdf"]


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::meta-reference",
            pip_packages=["faiss-cpu"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            deprecation_warning="Please use the `inline::faiss` provider instead.",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="Meta's reference implementation of a vector database.",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::faiss",
            pip_packages=["faiss-cpu"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.faiss",
            config_class="llama_stack.providers.inline.vector_io.faiss.FaissVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
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
""",
        ),
        # NOTE: sqlite-vec cannot be bundled into the container image because it does not have a
        # source distribution and the wheels are not available for all platforms.
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite-vec",
            pip_packages=["sqlite-vec"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.sqlite_vec",
            config_class="llama_stack.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
[SQLite-Vec](https://github.com/asg017/sqlite-vec) is an inline vector database provider for Llama Stack. It
allows you to store and query vectors directly within an SQLite database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features

- Lightweight and easy to use
- Fully integrated with Llama Stacks
- Uses disk-based storage for persistence, allowing for larger vector storage

### Comparison to Faiss

The choice between Faiss and sqlite-vec should be made based on the needs of your application,
as they have different strengths.

#### Choosing the Right Provider

Scenario | Recommended Tool | Reason
-- |-----------------| --
Online Analytical Processing (OLAP) | Faiss           | Fast, in-memory searches
Online Transaction Processing (OLTP) | sqlite-vec      | Frequent writes and reads
Frequent writes | sqlite-vec      | Efficient disk-based storage and incremental indexing
Large datasets | sqlite-vec      | Disk-based storage for larger vector storage
Datasets that can fit in memory, frequent reads | Faiss | Optimized for speed, indexing, and GPU acceleration

#### Empirical Example

Consider the histogram below in which 10,000 randomly generated strings were inserted
in batches of 100 into both Faiss and sqlite-vec using `client.tool_runtime.rag_tool.insert()`.

```{image} ../../../../_static/providers/vector_io/write_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

You will notice that the average write time for `sqlite-vec` was 788ms, compared to
47,640ms for Faiss. While the number is jarring, if you look at the distribution, you can see that it is rather
uniformly spread across the [1500, 100000] interval.

Looking at each individual write in the order that the documents are inserted you'll see the increase in
write speed as Faiss reindexes the vectors after each write.
```{image} ../../../../_static/providers/vector_io/write_time_sequence_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss write times
:width: 400px
```

In comparison, the read times for Faiss was on average 10% faster than sqlite-vec.
The modes of the two distributions highlight the differences much further where Faiss
will likely yield faster read performance.

```{image} ../../../../_static/providers/vector_io/read_time_comparison_sqlite-vec-faiss.png
:alt: Comparison of SQLite-Vec and Faiss read times
:width: 400px
```

## Usage

To use sqlite-vec in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use SQLite-Vec.
3. Start storing and querying vectors.

The SQLite-vec provider supports three search modes:

1. **Vector Search** (`mode="vector"`): Performs pure vector similarity search using the embeddings.
2. **Keyword Search** (`mode="keyword"`): Performs full-text search using SQLite's FTS5.
3. **Hybrid Search** (`mode="hybrid"`): Combines both vector and keyword search for better results. First performs keyword search to get candidate matches, then applies vector similarity search on those candidates.

Example with hybrid search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "hybrid", "max_chunks": 3, "score_threshold": 0.7},
)

# Using RRF ranker
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={
        "mode": "hybrid",
        "max_chunks": 3,
        "score_threshold": 0.7,
        "ranker": {"type": "rrf", "impact_factor": 60.0},
    },
)

# Using weighted ranker
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={
        "mode": "hybrid",
        "max_chunks": 3,
        "score_threshold": 0.7,
        "ranker": {"type": "weighted", "alpha": 0.7},  # 70% vector, 30% keyword
    },
)
```

Example with explicit vector search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "vector", "max_chunks": 3, "score_threshold": 0.7},
)
```

Example with keyword search:
```python
response = await vector_io.query_chunks(
    vector_store_id="my_db",
    query="your query here",
    params={"mode": "keyword", "max_chunks": 3, "score_threshold": 0.7},
)
```

## Supported Search Modes

The SQLite vector store supports three search modes:

1. **Vector Search** (`mode="vector"`): Uses vector similarity to find relevant chunks
2. **Keyword Search** (`mode="keyword"`): Uses keyword matching to find relevant chunks
3. **Hybrid Search** (`mode="hybrid"`): Combines both vector and keyword scores using a ranker

### Hybrid Search

Hybrid search combines the strengths of both vector and keyword search by:
- Computing vector similarity scores
- Computing keyword match scores
- Using a ranker to combine these scores

Two ranker types are supported:

1. **RRF (Reciprocal Rank Fusion)**:
   - Combines ranks from both vector and keyword results
   - Uses an impact factor (default: 60.0) to control the weight of higher-ranked results
   - Good for balancing between vector and keyword results
   - The default impact factor of 60.0 comes from the original RRF paper by Cormack et al. (2009) [^1], which found this value to provide optimal performance across various retrieval tasks

2. **Weighted**:
   - Linearly combines normalized vector and keyword scores
   - Uses an alpha parameter (0-1) to control the blend:
     - alpha=0: Only use keyword scores
     - alpha=1: Only use vector scores
     - alpha=0.5: Equal weight to both (default)

Example using RAGQueryConfig with different search modes:

```python
from llama_stack.apis.tools import RAGQueryConfig, RRFRanker, WeightedRanker

# Vector search
config = RAGQueryConfig(mode="vector", max_chunks=5)

# Keyword search
config = RAGQueryConfig(mode="keyword", max_chunks=5)

# Hybrid search with custom RRF ranker
config = RAGQueryConfig(
    mode="hybrid",
    max_chunks=5,
    ranker=RRFRanker(impact_factor=50.0),  # Custom impact factor
)

# Hybrid search with weighted ranker
config = RAGQueryConfig(
    mode="hybrid",
    max_chunks=5,
    ranker=WeightedRanker(alpha=0.7),  # 70% vector, 30% keyword
)

# Hybrid search with default RRF ranker
config = RAGQueryConfig(
    mode="hybrid", max_chunks=5
)  # Will use RRF with impact_factor=60.0
```

Note: The ranker configuration is only used in hybrid mode. For vector or keyword modes, the ranker parameter is ignored.

## Installation

You can install SQLite-Vec using pip:

```bash
pip install sqlite-vec
```

## Documentation

See [sqlite-vec's GitHub repo](https://github.com/asg017/sqlite-vec/tree/main) for more details about sqlite-vec in general.

[^1]: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114). In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (pp. 758-759).
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::sqlite_vec",
            pip_packages=["sqlite-vec"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.sqlite_vec",
            config_class="llama_stack.providers.inline.vector_io.sqlite_vec.SQLiteVectorIOConfig",
            deprecation_warning="Please use the `inline::sqlite-vec` provider (notice the hyphen instead of underscore) instead.",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
Please refer to the sqlite-vec provider documentation.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="chromadb",
            provider_type="remote::chromadb",
            pip_packages=["chromadb-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.remote.vector_io.chroma",
            config_class="llama_stack.providers.remote.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for Llama Stack. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::chromadb",
            pip_packages=["chromadb"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.chroma",
            config_class="llama_stack.providers.inline.vector_io.chroma.ChromaVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
[Chroma](https://www.trychroma.com/) is an inline and remote vector
database provider for Llama Stack. It allows you to store and query vectors directly within a Chroma database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features
Chroma supports:
- Store embeddings and their metadata
- Vector search
- Full-text search
- Document storage
- Metadata filtering
- Multi-modal retrieval

## Usage

To use Chrome in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use chroma.
3. Start storing and querying vectors.

## Installation

You can install chroma using pip:

```bash
pip install chromadb
```

## Documentation
See [Chroma's documentation](https://docs.trychroma.com/docs/overview/introduction) for more details about Chroma in general.

""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="pgvector",
            provider_type="remote::pgvector",
            pip_packages=["psycopg2-binary"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.remote.vector_io.pgvector",
            config_class="llama_stack.providers.remote.vector_io.pgvector.PGVectorVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
[PGVector](https://github.com/pgvector/pgvector) is a remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with Llama Stack

There are three implementations of search for PGVectoIndex available:

1. Vector Search:
- How it works:
  - Uses PostgreSQL's vector extension (pgvector) to perform similarity search
  - Compares query embeddings against stored embeddings using Cosine distance or other distance metrics
  - Eg. SQL query: SELECT document, embedding &lt;=&gt; %s::vector AS distance FROM table ORDER BY distance

-Characteristics:
  - Semantic understanding - finds documents similar in meaning even if they don't share keywords
  - Works with high-dimensional vector embeddings (typically 768, 1024, or higher dimensions)
  - Best for: Finding conceptually related content, handling synonyms, cross-language search

2. Keyword Search
- How it works:
  - Uses PostgreSQL's full-text search capabilities with tsvector and ts_rank
  - Converts text to searchable tokens using to_tsvector('english', text). Default language is English.
  - Eg. SQL query: SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score

- Characteristics:
  - Lexical matching - finds exact keyword matches and variations
  - Uses GIN (Generalized Inverted Index) for fast text search performance
  - Scoring: Uses PostgreSQL's ts_rank function for relevance scoring
  - Best for: Exact term matching, proper names, technical terms, Boolean-style queries

3. Hybrid Search
- How it works:
  - Combines both vector and keyword search results
  - Runs both searches independently, then merges results using configurable reranking

- Two reranking strategies available:
    - Reciprocal Rank Fusion (RRF) - (default: 60.0)
    - Weighted Average - (default: 0.5)

- Characteristics:
  - Best of both worlds: semantic understanding + exact matching
  - Documents appearing in both searches get boosted scores
  - Configurable balance between semantic and lexical matching
  - Best for: General-purpose search where you want both precision and recall

4. Database Schema
The PGVector implementation stores data optimized for all three search types:
CREATE TABLE vector_store_xxx (
    id TEXT PRIMARY KEY,
    document JSONB,                    -- Original document
    embedding vector(dimension),        -- For vector search
    content_text TEXT,                 -- Raw text content
    tokenized_content TSVECTOR          -- For keyword search
);

-- Indexes for performance
CREATE INDEX content_gin_idx ON table USING GIN(tokenized_content);  -- Keyword search
-- Vector index created automatically by pgvector

## Usage

To use PGVector in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use pgvector. (e.g. remote::pgvector).
3. Start storing and querying vectors.

## This is an example how you can set up your environment for using PGVector

1. Export env vars:
```bash
export ENABLE_PGVECTOR=true
export PGVECTOR_HOST=localhost
export PGVECTOR_PORT=5432
export PGVECTOR_DB=llamastack
export PGVECTOR_USER=llamastack
export PGVECTOR_PASSWORD=llamastack
```

2. Create DB:
```bash
psql -h localhost -U postgres -c "CREATE ROLE llamastack LOGIN PASSWORD 'llamastack';"
psql -h localhost -U postgres -c "CREATE DATABASE llamastack OWNER llamastack;"
psql -h localhost -U llamastack -d llamastack -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Installation

You can install PGVector using docker:

```bash
docker pull pgvector/pgvector:pg17
```
## Documentation
See [PGVector's documentation](https://github.com/pgvector/pgvector) for more details about PGVector in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="weaviate",
            provider_type="remote::weaviate",
            pip_packages=["weaviate-client>=4.16.5"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.remote.vector_io.weaviate",
            config_class="llama_stack.providers.remote.vector_io.weaviate.WeaviateVectorIOConfig",
            provider_data_validator="llama_stack.providers.remote.vector_io.weaviate.WeaviateRequestProviderData",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
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
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::qdrant",
            pip_packages=["qdrant-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.qdrant",
            config_class="llama_stack.providers.inline.vector_io.qdrant.QdrantVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description=r"""
[Qdrant](https://qdrant.tech/documentation/) is an inline and remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

> By default, Qdrant stores vectors in RAM, delivering incredibly fast access for datasets that fit comfortably in
> memory. But when your dataset exceeds RAM capacity, Qdrant offers Memmap as an alternative.
>
> \[[An Introduction to Vector Databases](https://qdrant.tech/articles/what-is-a-vector-database/)\]



## Features

- Lightweight and easy to use
- Fully integrated with Llama Stack
- Apache 2.0 license terms
- Store embeddings and their metadata
- Supports search by
  [Keyword](https://qdrant.tech/articles/qdrant-introduces-full-text-filters-and-indexes/)
  and [Hybrid](https://qdrant.tech/articles/hybrid-search/#building-a-hybrid-search-system-in-qdrant) search
- [Multilingual and Multimodal retrieval](https://qdrant.tech/documentation/multimodal-search/)
- [Medatata filtering](https://qdrant.tech/articles/vector-search-filtering/)
- [GPU support](https://qdrant.tech/documentation/guides/running-with-gpu/)

## Usage

To use Qdrant in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use Qdrant.
3. Start storing and querying vectors.

## Installation

You can install Qdrant using docker:

```bash
docker pull qdrant/qdrant
```
## Documentation
See the [Qdrant documentation](https://qdrant.tech/documentation/) for more details about Qdrant in general.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="qdrant",
            provider_type="remote::qdrant",
            pip_packages=["qdrant-client"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.remote.vector_io.qdrant",
            config_class="llama_stack.providers.remote.vector_io.qdrant.QdrantVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
Please refer to the inline provider documentation.
""",
        ),
        RemoteProviderSpec(
            api=Api.vector_io,
            adapter_type="milvus",
            provider_type="remote::milvus",
            pip_packages=["pymilvus>=2.4.10"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.remote.vector_io.milvus",
            config_class="llama_stack.providers.remote.vector_io.milvus.MilvusVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
[Milvus](https://milvus.io/) is an inline and remote vector database provider for Llama Stack. It
allows you to store and query vectors directly within a Milvus database.
That means you're not limited to storing vectors in memory or in a separate service.

## Features

- Easy to use
- Fully integrated with Llama Stack
- Supports all search modes: vector, keyword, and hybrid search (both inline and remote configurations)

## Usage

To use Milvus in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use Milvus.
3. Start storing and querying vectors.

## Installation

If you want to use inline Milvus, you can install:

```bash
pip install pymilvus[milvus-lite]
```

If you want to use remote Milvus, you can install:

```bash
pip install pymilvus
```

## Configuration

In Llama Stack, Milvus can be configured in two ways:
- **Inline (Local) Configuration** - Uses Milvus-Lite for local storage
- **Remote Configuration** - Connects to a remote Milvus server

### Inline (Local) Configuration

The simplest method is local configuration, which requires setting `db_path`, a path for locally storing Milvus-Lite files:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: inline::milvus
    config:
      db_path: ~/.llama/distributions/together/milvus_store.db
```

### Remote Configuration

Remote configuration is suitable for larger data storage requirements:

#### Standard Remote Connection

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "http://<host>:<port>"
      token: "<user>:<password>"
```

#### TLS-Enabled Remote Connection (One-way TLS)

For connections to Milvus instances with one-way TLS enabled:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "https://<host>:<port>"
      token: "<user>:<password>"
      secure: True
      server_pem_path: "/path/to/server.pem"
```

#### Mutual TLS (mTLS) Remote Connection

For connections to Milvus instances with mutual TLS (mTLS) enabled:

```yaml
vector_io:
  - provider_id: milvus
    provider_type: remote::milvus
    config:
      uri: "https://<host>:<port>"
      token: "<user>:<password>"
      secure: True
      ca_pem_path: "/path/to/ca.pem"
      client_pem_path: "/path/to/client.pem"
      client_key_path: "/path/to/client.key"
```

#### Key Parameters for TLS Configuration

- **`secure`**: Enables TLS encryption when set to `true`. Defaults to `false`.
- **`server_pem_path`**: Path to the **server certificate** for verifying the server's identity (used in one-way TLS).
- **`ca_pem_path`**: Path to the **Certificate Authority (CA) certificate** for validating the server certificate (required in mTLS).
- **`client_pem_path`**: Path to the **client certificate** file (required for mTLS).
- **`client_key_path`**: Path to the **client private key** file (required for mTLS).

## Search Modes

Milvus supports three different search modes for both inline and remote configurations:

### Vector Search
Vector search uses semantic similarity to find the most relevant chunks based on embedding vectors. This is the default search mode and works well for finding conceptually similar content.

```python
# Vector search example
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="What is machine learning?",
    search_mode="vector",
    max_num_results=5,
)
```

### Keyword Search
Keyword search uses traditional text-based matching to find chunks containing specific terms or phrases. This is useful when you need exact term matches.

```python
# Keyword search example
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="Python programming language",
    search_mode="keyword",
    max_num_results=5,
)
```

### Hybrid Search
Hybrid search combines both vector and keyword search methods to provide more comprehensive results. It leverages the strengths of both semantic similarity and exact term matching.

#### Basic Hybrid Search
```python
# Basic hybrid search example (uses RRF ranker with default impact_factor=60.0)
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
)
```

**Note**: The default `impact_factor` value of 60.0 was empirically determined to be optimal in the original RRF research paper: ["Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack et al., 2009).

#### Hybrid Search with RRF (Reciprocal Rank Fusion) Ranker
RRF combines rankings from vector and keyword search by using reciprocal ranks. The impact factor controls how much weight is given to higher-ranked results.

```python
# Hybrid search with custom RRF parameters
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
    ranking_options={
        "ranker": {
            "type": "rrf",
            "impact_factor": 100.0,  # Higher values give more weight to top-ranked results
        }
    },
)
```

#### Hybrid Search with Weighted Ranker
Weighted ranker linearly combines normalized scores from vector and keyword search. The alpha parameter controls the balance between the two search methods.

```python
# Hybrid search with weighted ranker
search_response = client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks in Python",
    search_mode="hybrid",
    max_num_results=5,
    ranking_options={
        "ranker": {
            "type": "weighted",
            "alpha": 0.7,  # 70% vector search, 30% keyword search
        }
    },
)
```

For detailed documentation on RRF and Weighted rankers, please refer to the [Milvus Reranking Guide](https://milvus.io/docs/reranking.md).

## Documentation
See the [Milvus documentation](https://milvus.io/docs/install-overview.md) for more details about Milvus in general.

For more details on TLS configuration, refer to the [TLS setup guide](https://milvus.io/docs/tls.md).
""",
        ),
        InlineProviderSpec(
            api=Api.vector_io,
            provider_type="inline::milvus",
            pip_packages=["pymilvus[milvus-lite]>=2.4.10"] + DEFAULT_VECTOR_IO_DEPS,
            module="llama_stack.providers.inline.vector_io.milvus",
            config_class="llama_stack.providers.inline.vector_io.milvus.MilvusVectorIOConfig",
            api_dependencies=[Api.inference],
            optional_api_dependencies=[Api.files, Api.models],
            description="""
Please refer to the remote provider documentation.
""",
        ),
    ]
