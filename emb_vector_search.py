#!/usr/bin/env python3
"""
Loads the embedding model (all-MiniLM-L6-v2)
Creates a k-NN enabled index
Indexes markdown documents from a repository with their embeddings
"""

import os
import sys
import argparse
from pathlib import Path
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


def find_markdown_files(repo_path: str) -> List[str]:
    """
    Recursively find all markdown files in a repository
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of paths to markdown files
    """
    md_files = []
    repo_path = Path(repo_path).resolve()
    
    if not repo_path.exists():
        print(f"❌ Error: Repository path does not exist: {repo_path}")
        return []
    
    print(f"📁 Scanning repository: {repo_path}")
    
    for file_path in repo_path.rglob("*.md"):
        # Skip hidden directories and common excludes
        if any(part.startswith('.') for part in file_path.parts):
            continue
        if any(exclude in str(file_path) for exclude in ['node_modules', 'venv', 'env', '.git']):
            continue
        
        md_files.append(str(file_path))
    
    print(f"📄 Found {len(md_files)} markdown files")
    return md_files


def read_markdown_file(file_path: str) -> Dict[str, str]:
    """
    Read a markdown file and extract title and content
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        Dictionary with title, text, metadata, and file_path
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from first h1 heading or use filename
        lines = content.split('\n')
        title = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        if not title:
            title = Path(file_path).stem.replace('-', ' ').replace('_', ' ').title()
        
        # Get relative path for metadata
        rel_path = file_path
        
        return {
            'title': title,
            'text': content,
            'metadata': rel_path,
            'file_path': file_path
        }
    
    except Exception as e:
        print(f"⚠️  Error reading {file_path}: {e}")
        return None


def load_documents_from_repo(repo_path: str) -> List[Dict[str, str]]:
    """
    Load all markdown documents from a repository
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        List of document dictionaries
    """
    md_files = find_markdown_files(repo_path)
    
    if not md_files:
        print("⚠️  No markdown files found in the repository")
        return []
    
    documents = []
    print("\n📖 Reading markdown files...")
    
    for file_path in md_files:
        doc = read_markdown_file(file_path)
        if doc:
            documents.append(doc)
            print(f"   ✓ {Path(file_path).name}")
    
    print(f"\n✅ Successfully loaded {len(documents)} documents")
    return documents


def main():
    """Main function to index markdown documents from a repository"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Index markdown documents from a repository into OpenSearch vector database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index markdown files from a repository
  ./emb_vector_search.py /path/to/repository
  
  # Index from current directory
  ./emb_vector_search.py .
  
  # Specify custom index name
  ./emb_vector_search.py /path/to/repo --index my-docs
        """
    )
    
    parser.add_argument(
        'repo_path',
        type=str,
        help='Path to the repository containing markdown files'
    )
    
    parser.add_argument(
        '--index',
        type=str,
        default='documents',
        help='Name of the OpenSearch index (default: documents)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='OpenSearch host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=9200,
        help='OpenSearch port (default: 9200)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model name (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 Vector Database Indexer")
    print("="*80)
    
    # Load documents from repository
    documents = load_documents_from_repo(args.repo_path)
    
    if not documents:
        print("\n❌ No documents to index. Exiting.")
        sys.exit(1)
    
    # Initialize the vector database
    print("\n" + "="*80)
    engine = VectorDatabase(
        opensearch_host=args.host,
        opensearch_port=args.port,
        model_name=args.model
    )
    
    # Create index
    print("\n" + "="*80)
    engine.create_index(index_name=args.index)
    
    # Index documents
    print("\n" + "="*80)
    engine.index_documents(documents, index_name=args.index)
    
    print("\n" + "="*80)
    print("✅ Vector database setup completed!")
    print(f"📊 Total documents indexed: {len(documents)}")
    print(f"📍 Index name: {args.index}")
    print(f"🔢 Embedding dimension: {engine.embedding_dim}")
    print("="*80)


if __name__ == "__main__":
    main()
