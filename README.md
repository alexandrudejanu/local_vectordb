# OpenSearch Vector Database

A local vector database setup using OpenSearch with k-NN plugin for semantic search and similarity matching.

## Features and  Prerequisites

- **Vector Search**: Built-in k-NN plugin with HNSW and IVF algorithms
- **Local Embeddings**: Sentence-transformers for generating text embeddings locally
- **OpenSearch Dashboards**: Web UI for visualization and management
- **Persistent Storage**: Docker volumes for data persistence
- **Apple Silicon Optimized**: Works great on M1/M2/M3/M4 chips

- Docker and Docker Compose installed
- Python 3.8+ (for Python client)
- At least 2GB of available RAM

## Quick Start

###  Start the Services

```bash
# OpenSearch on port 9200
# OpenSearch Dashboards on port 5601
docker-compose up -d
```

###  Verify Services are Running

```bash
# Check OpenSearch
curl http://localhost:9200

# Check cluster health
curl http://localhost:9200/_cluster/health?pretty

# Check indices
GET _cat/indices?v&pretty
```

### Load the sentence-transformer model and index documents

```bash
pip install -r requirements.txt

# Index markdown files from a repository
./emb_vector_search.py /path/to/repository

# Examples:
./emb_vector_search.py ~/Documents/my-notes
./emb_vector_search.py . --index my-docs
./emb_vector_search.py /path/to/repo --model all-mpnet-base-v2
```

This script:
- Recursively finds all `.md` files in the specified repository
- Loads a local embeddings model (all-MiniLM-L6-v2 by default)
- Creates a vector index in OpenSearch
- Indexes all markdown documents with their embeddings

**Options:**
- `--index NAME` - Custom index name (default: documents)
- `--host HOST` - OpenSearch host (default: localhost)
- `--port PORT` - OpenSearch port (default: 9200)
- `--model MODEL` - Embedding model name (default: all-MiniLM-L6-v2)

### Search the Vector Database

```bash
# Interactive search mode (uses default 'documents' index)
./search.py

# Interactive search with custom index
./search.py --index observability

# Or search directly from command line
./search.py "artificial intelligence"
./search.py --index observability "kubernetes deployment"

# Search with more results
./search.py --index observability --k 10 "docker configuration"
```

**Search Options:**
- `--index NAME` - Index name to search (default: documents)
- `--host HOST` - OpenSearch host (default: localhost)
- `--port PORT` - OpenSearch port (default: 9200)
- `--model MODEL` - Embedding model name (default: all-MiniLM-L6-v2)
- `--k N` - Number of results to return (default: 5)

The search script provides:
- Semantic similarity search using vector embeddings
- Interactive or command-line search modes
- Formatted results with relevance scores and file paths
- Configurable number of results

## Complete Workflow Example

```bash
# 1. Start OpenSearch
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Index your markdown documentation
./emb_vector_search.py ~/my-project/docs --index my-docs

# 4. Search your docs
./search.py --index my-docs "how to configure authentication"

# Or use interactive mode with your custom index
./search.py --index my-docs
```

## Local Embeddings Models

The setup uses **sentence-transformers** for generating text embeddings locally. No API keys or internet connection needed after initial model download!

### Recommended Models for English Text

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ Fast | Good | General purpose (recommended) |
| `all-mpnet-base-v2` | 768 | ⚡⚡ Medium | Better | Higher quality needed |
| `bge-small-en-v1.5` | 384 | ⚡⚡⚡ Fast | Good | General purpose |
| `bge-base-en-v1.5` | 768 | ⚡⚡ Medium | Better | Balanced quality/speed |

## Performance Tips

1. **Dimension Size**: Lower dimensions = faster search, but may reduce accuracy
2. **ef_construction**: Higher values = better recall, slower indexing
3. **ef_search**: Higher values = better recall, slower queries
4. **m parameter**: Higher values = better recall, more memory usage

## Resources

- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [k-NN Plugin Guide](https://opensearch.org/docs/latest/search-plugins/knn/index/)
- [OpenSearch Python Client](https://github.com/opensearch-project/opensearch-py)

