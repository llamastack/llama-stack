# File Operations and Vector Store Integration

## Overview

Llama Stack provides seamless integration between the Files API and Vector Store APIs, enabling you to upload documents and automatically process them into searchable vector embeddings. This integration implements file operations following the [OpenAI Vector Store Files API specification](https://platform.openai.com/docs/api-reference/vector-stores-files).

## Enhanced Capabilities Beyond OpenAI

While Llama Stack maintains full compatibility with OpenAI's Vector Store API, it provides several additional capabilities that enhance functionality and flexibility:

### **Embedding Model Specification**
Unlike OpenAI's vector stores which use a fixed embedding model, Llama Stack allows you to specify which embedding model to use when creating a vector store:

```python
# Create vector store with specific embedding model
vector_store = client.vector_stores.create(
    name="my_documents",
    embedding_model="all-MiniLM-L6-v2",  # Specify your preferred model
    embedding_dimension=384,
)
```

### **Advanced Search Modes**
Llama Stack supports multiple search modes beyond basic vector similarity:

- **Vector Search**: Pure semantic similarity search using embeddings
- **Keyword Search**: Traditional keyword-based search for exact matches
- **Hybrid Search**: Combines both vector and keyword search for optimal results

```python
# Different search modes
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="machine learning algorithms",
    search_mode="hybrid",  # or "vector", "keyword"
    max_num_results=5,
)
```

### **Flexible Ranking Options**
For hybrid search, Llama Stack offers configurable ranking strategies:

- **RRF (Reciprocal Rank Fusion)**: Combines rankings with configurable impact factor
- **Weighted Ranker**: Linear combination of vector and keyword scores with adjustable weights

```python
# Custom ranking configuration
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="neural networks",
    search_mode="hybrid",
    ranking_options={
        "ranker": {"type": "weighted", "alpha": 0.7}  # 70% vector, 30% keyword
    },
)
```

### **Provider Selection**
Choose from multiple vector store providers based on your specific needs:

- **Inline Providers**: FAISS (fast in-memory), SQLite-vec (disk-based), Milvus (high-performance)
- **Remote Providers**: ChromaDB, Qdrant, Weaviate, Postgres (PGVector), Milvus

```python
# Specify provider when creating vector store
vector_store = client.vector_stores.create(
    name="my_documents", provider_id="sqlite-vec"  # Choose your preferred provider
)
```

## How It Works

The file operations work through several key components:

1. **File Upload**: Documents are uploaded through the Files API
2. **Automatic Processing**: Files are automatically chunked and converted to embeddings
3. **Vector Storage**: Chunks are stored in vector databases with metadata
4. **Search & Retrieval**: Users can search through processed documents using natural language

## Supported Vector Store Providers

The following vector store providers support file operations:

### Inline Providers (Single Node)

- **FAISS**: Fast in-memory vector similarity search
- **SQLite-vec**: Disk-based storage with hybrid search capabilities
- **Milvus**: High-performance vector database with advanced indexing

### Remote Providers (Hosted)

- **ChromaDB**: Vector database with metadata filtering
- **Qdrant**: Vector similarity search with payload filtering
- **Weaviate**: Vector database with GraphQL interface
- **Postgres (PGVector)**: Vector extensions for PostgreSQL

## File Processing Pipeline

### 1. File Upload

```python
from llama_stack import LlamaStackClient

client = LlamaStackClient("http://localhost:8000")

# Upload a document
with open("document.pdf", "rb") as f:
    file_info = await client.files.upload(file=f, purpose="assistants")
```

### 2. Attach to Vector Store

```python
# Create a vector store
vector_store = client.vector_stores.create(name="my_documents")

# Attach the file to the vector store
file_attach_response = await client.vector_stores.files.create(
    vector_store_id=vector_store.id, file_id=file_info.id
)
```

### 3. Automatic Processing

The system automatically:
- Detects the file type and extracts text content
- Splits content into chunks (default: 800 tokens with 400 token overlap)
- Generates embeddings for each chunk
- Stores chunks with metadata in the vector store
- Updates file status to "completed"

### 4. Search and Retrieval

```python
# Search through processed documents
search_results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="What is the main topic discussed?",
    max_num_results=5,
)

# Process results
for result in search_results.data:
    print(f"Score: {result.score}")
    for content in result.content:
        print(f"Content: {content.text}")
```

## Supported File Types

The FileResponse system supports various document formats:

- **Text Files**: `.txt`, `.md`, `.rst`
- **Documents**: `.pdf`, `.docx`, `.doc`
- **Code**: `.py`, `.js`, `.java`, `.cpp`, etc.
- **Data**: `.json`, `.csv`, `.xml`
- **Web Content**: HTML files

## Chunking Strategies

### Default Strategy

The default chunking strategy uses:
- **Max Chunk Size**: 800 tokens
- **Overlap**: 400 tokens
- **Method**: Semantic boundary detection

### Custom Chunking

You can customize chunking when attaching files:

```python
from llama_stack.apis.vector_io import VectorStoreChunkingStrategy

# Custom chunking strategy
chunking_strategy = VectorStoreChunkingStrategy(
    type="custom", max_chunk_size_tokens=1000, chunk_overlap_tokens=200
)

# Attach file with custom chunking
file_attach_response = await client.vector_stores.files.create(
    vector_store_id=vector_store.id,
    file_id=file_info.id,
    chunking_strategy=chunking_strategy,
)
```

**Note**: While Llama Stack is OpenAI-compatible, it also supports additional options beyond the standard OpenAI API. When creating vector stores, you can specify custom embedding models and embedding dimensions that will be used when processing chunks from attached files.


## File Management

### List Files in Vector Store

```python
# List all files in a vector store
files = await client.vector_stores.files.list(vector_store_id=vector_store.id)

for file in files:
    print(f"File: {file.filename}, Status: {file.status}")
```

### File Status Tracking

Files go through several statuses:
- **in_progress**: File is being processed
- **completed**: File successfully processed and searchable
- **failed**: Processing failed (check `last_error` for details)
- **cancelled**: Processing was cancelled

### Retrieve File Content

```python
# Get chunked content from vector store
content_response = await client.vector_stores.files.retrieve_content(
    vector_store_id=vector_store.id, file_id=file_info.id
)

for chunk in content_response.content:
    print(f"Chunk {chunk.metadata.get('chunk_index', 0)}: {chunk.text}")
```

## Vector Store Management

### List Vector Stores

Retrieve a paginated list of all vector stores:

```python
# List all vector stores with default pagination
vector_stores = await client.vector_stores.list()

# Custom pagination and ordering
vector_stores = await client.vector_stores.list(
    limit=10,
    order="asc",  # or "desc"
    after="vs_12345678",  # cursor-based pagination
)

for store in vector_stores.data:
    print(f"Store: {store.name}, Files: {store.file_counts.total}")
    print(f"Created: {store.created_at}, Status: {store.status}")
```

### Retrieve Vector Store Details

Get detailed information about a specific vector store:

```python
# Get vector store details
store_details = await client.vector_stores.retrieve(vector_store_id="vs_12345678")

print(f"Name: {store_details.name}")
print(f"Status: {store_details.status}")
print(f"File Counts: {store_details.file_counts}")
print(f"Usage: {store_details.usage_bytes} bytes")
print(f"Created: {store_details.created_at}")
print(f"Metadata: {store_details.metadata}")
```

### Update Vector Store

Modify vector store properties such as name, metadata, or expiration settings:

```python
# Update vector store name and metadata
updated_store = await client.vector_stores.update(
    vector_store_id="vs_12345678",
    name="Updated Document Collection",
    metadata={
        "description": "Updated collection for research",
        "category": "research",
        "version": "2.0",
    },
)

# Set expiration policy
expired_store = await client.vector_stores.update(
    vector_store_id="vs_12345678",
    expires_after={"anchor": "last_active_at", "days": 30},
)

print(f"Updated store: {updated_store.name}")
print(f"Last active: {updated_store.last_active_at}")
```

### Delete Vector Store

Remove a vector store and all its associated data:

```python
# Delete a vector store
delete_response = await client.vector_stores.delete(vector_store_id="vs_12345678")

if delete_response.deleted:
    print(f"Vector store {delete_response.id} successfully deleted")
else:
    print("Failed to delete vector store")
```

**Important Notes:**
- Deleting a vector store removes all files, chunks, and embeddings
- This operation cannot be undone
- The underlying vector database is also cleaned up
- Consider backing up important data before deletion

## Search Capabilities

### Vector Search

Pure similarity search using embeddings:

```python
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="machine learning algorithms",
    max_num_results=10,
)
```

### Filtered Search

Combine vector search with metadata filtering:

```python
results = await client.vector_stores.search(
    vector_store_id=vector_store.id,
    query="machine learning algorithms",
    filters={"file_type": "pdf", "upload_date": "2024-01-01"},
    max_num_results=10,
)
```

### Hybrid Search

[SQLite-vec](../providers/vector_io/inline_sqlite-vec.md), [pgvector](../providers/vector_io/remote_pgvector.md), and [Milvus](../providers/vector_io/inline_milvus.md) support combining vector and keyword search.

## Performance Considerations

> **Note**: For detailed performance optimization strategies, see [Performance Considerations](../providers/files/openai_file_operations_support.md#performance-considerations) in the provider documentation.

**Key Points:**
- **Chunk Size**: 400-600 tokens for precision, 800-1200 for context
- **Storage**: Choose provider based on your performance needs
- **Search**: Optimize for your specific use case

## Error Handling

> **Note**: For comprehensive troubleshooting and error handling, see [Troubleshooting](../providers/files/openai_file_operations_support.md#troubleshooting) in the provider documentation.

**Common Issues:**
- File processing failures (format, size limits)
- Search performance optimization
- Storage and memory issues

## Best Practices

> **Note**: For detailed best practices and recommendations, see [Best Practices](../providers/files/openai_file_operations_support.md#best-practices) in the provider documentation.

**Key Recommendations:**
- File organization and naming conventions
- Chunking strategy optimization
- Metadata and monitoring practices
- Regular cleanup and maintenance

## Integration Examples

### RAG Application

```python
# Build a RAG system with file uploads
async def build_rag_system():
    # Create vector store
    vector_store = client.vector_stores.create(name="knowledge_base")

    # Upload and process documents
    documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    for doc in documents:
        with open(doc, "rb") as f:
            file_info = await client.files.create(file=f, purpose="assistants")
            await client.vector_stores.files.create(
                vector_store_id=vector_store.id, file_id=file_info.id
            )

    return vector_store


# Query the RAG system
async def query_rag(vector_store_id, question):
    results = await client.vector_stores.search(
        vector_store_id=vector_store_id, query=question, max_num_results=5
    )
    return results
```

### Document Analysis

```python
# Analyze document content through vector search
async def analyze_document(vector_store_id, file_id):
    # Get document content
    content = await client.vector_stores.files.retrieve_content(
        vector_store_id=vector_store_id, file_id=file_id
    )

    # Search for specific topics
    topics = ["introduction", "methodology", "conclusion"]
    analysis = {}

    for topic in topics:
        results = await client.vector_stores.search(
            vector_store_id=vector_store_id, query=topic, max_num_results=3
        )
        analysis[topic] = results.data

    return analysis
```

## Next Steps

- Explore the [Files API documentation](../apis/files.md) for detailed API reference
- Check [Vector Store Providers](../providers/vector_io/index.md) for specific implementation details
- Review [Getting Started](../getting_started/index.md) for quick setup instructions
