#!/usr/bin/env python3
"""
Loads the embedding model (all-MiniLM-L6-v2)
Creates a k-NN enabled index
Indexes 8 documents with their embeddings
"""

from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from typing import List, Dict


class VectorDatabase:
    """Local vector database using OpenSearch and sentence-transformers"""
    
    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector database
        
        Args:
            opensearch_host: OpenSearch host
            opensearch_port: OpenSearch port
            model_name: Name of the sentence-transformer model to use
                       Options:
                       - 'all-MiniLM-L6-v2': Fast, 384 dims (recommended for starting)
                       - 'all-mpnet-base-v2': Better quality, 768 dims
                       - 'bge-small-en-v1.5': Good balance, 384 dims
        """
        print(f"🔌 Connecting to OpenSearch at {opensearch_host}:{opensearch_port}")
        self.client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        
        print(f"🤖 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def create_index(self, index_name: str = "documents"):
        """Create a k-NN enabled index for vector search"""
        
        # Delete index if it exists
        if self.client.indices.exists(index=index_name):
            print(f"🗑️  Deleting existing index: {index_name}")
            self.client.indices.delete(index=index_name)
        
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                }
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "text": {
                        "type": "text"
                    },
                    "title": {
                        "type": "text"
                    },
                    "metadata": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        self.client.indices.create(index=index_name, body=index_body)
        print(f"✅ Created index: {index_name} with {self.embedding_dim} dimensions")
    
    def index_documents(self, documents: List[Dict[str, str]], index_name: str = "documents"):
        """
        Index documents with embeddings
        
        Args:
            documents: List of dicts with 'text', optional 'title' and 'metadata'
            index_name: Name of the index
        """
        print(f"📝 Indexing {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            # Generate embedding
            text_to_embed = doc.get('text', '')
            embedding = self.model.encode(text_to_embed).tolist()
            
            # Prepare document
            doc_body = {
                'embedding': embedding,
                'text': doc.get('text', ''),
                'title': doc.get('title', ''),
                'metadata': doc.get('metadata', '')
            }
            
            # Index document
            self.client.index(
                index=index_name,
                id=i,
                body=doc_body,
                refresh=True
            )
        
        print(f"✅ Indexed {len(documents)} documents")


def main():
    """Example usage"""
    
    # Initialize the vector database
    engine = VectorDatabase(
        opensearch_host="localhost",
        opensearch_port=9200,
        model_name="all-MiniLM-L6-v2"  # Fast and efficient for Apple Silicon
    )
    
    # Create index
    engine.create_index(index_name="documents")
    
    # Sample documents
    documents = [
        {
            "title": "Python Programming",
            "text": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "metadata": "programming"
        },
        {
            "title": "Machine Learning",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "metadata": "ai"
        },
        {
            "title": "Vector Databases",
            "text": "Vector databases are specialized systems designed to store and query high-dimensional vectors efficiently.",
            "metadata": "database"
        },
        {
            "title": "Natural Language Processing",
            "text": "NLP is a field of AI that focuses on the interaction between computers and human language.",
            "metadata": "ai"
        },
        {
            "title": "Web Development",
            "text": "Web development involves building and maintaining websites using technologies like HTML, CSS, and JavaScript.",
            "metadata": "programming"
        },
        {
            "title": "Data Science",
            "text": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
            "metadata": "data"
        },
        {
            "title": "Cloud Computing",
            "text": "Cloud computing delivers computing services over the internet, including storage and processing power.",
            "metadata": "infrastructure"
        },
        {
            "title": "Deep Learning",
            "text": "Deep learning uses neural networks with multiple layers to learn complex patterns in large datasets.",
            "metadata": "ai"
        }
    ]
    
    # Index documents
    engine.index_documents(documents, index_name="documents")
    
    print("\n" + "="*80)
    print("✅ Vector database setup completed!")
    print(f"📊 Total documents indexed: {len(documents)}")
    print(f"📍 Index name: documents")
    print(f"🔢 Embedding dimension: {engine.embedding_dim}")
    print("="*80)


if __name__ == "__main__":
    main()
