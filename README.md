# OpenSearch Vector Database

A local vector database setup using OpenSearch with k-NN plugin for semantic search and similarity matching.

## Features

- **Vector Search**: Built-in k-NN plugin with HNSW and IVF algorithms
- **Local Embeddings**: Sentence-transformers for generating text embeddings locally
- **OpenSearch Dashboards**: Web UI for visualization and management
- **Persistent Storage**: Docker volumes for data persistence
- **Apple Silicon Optimized**: Works great on M1/M2/M3/M4 chips

## Prerequisites

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
./emb_vector_search.py
```

This example script demonstrates:
- Loading a local embeddings model (all-MiniLM-L6-v2)
- Creating a vector index
- Indexing documents with embeddings

### Search the Vector Database

```bash
# Interactive search mode
python search.py

# Or search directly from command line
python search.py "artificial intelligence"
python search.py "programming languages"
```

The search script provides:
- Semantic similarity search using vector embeddings
- Interactive or command-line search modes
- Formatted results with relevance scores
- Top 5 most similar documents returned

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

