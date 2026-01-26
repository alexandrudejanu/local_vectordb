#!/usr/bin/env python3
"""
Search the vector database using semantic similarity
"""

import argparse
import sys
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from typing import List, Dict


class VectorSearch:
    """Search engine for the vector database"""
    
    def __init__(
        self,
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """Initialize search engine with embedding model and OpenSearch client"""
        print(f"Connecting to OpenSearch at {opensearch_host}:{opensearch_port}")
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
        print(f"Ready to search!\n")
    
    def search(
        self,
        query: str,
        index_name: str = "documents",
        k: int = 5
    ) -> List[Dict]:
        """
        Perform semantic similarity search
        
        Args:
            query: Search query text
            index_name: Name of the index to search
            k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Looking for top {k} results...\n")
        
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        # k-NN search
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k
                    }
                }
            }
        }
        
        response = self.client.search(index=index_name, body=search_body)
        
        # Format results
        results = []
        for hit in response['hits']['hits']:
            results.append({
                'score': hit['_score'],
                'title': hit['_source'].get('title', ''),
                'text': hit['_source']['text'],
                'metadata': hit['_source'].get('metadata', ''),
                'file_path': hit['_source'].get('file_path', '')
            })
        
        return results
    
    def display_results(self, results: List[Dict]):
        """Display search results in a formatted way"""
        if not results:
            print("❌ No results found.")
            return
        
        print("="*80)
        print(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   Score: {result['score']:.4f}")
            
            # Show file path if it's a file-based document
            if result.get('file_path'):
                print(f"   File: {result['file_path']}")
            elif result.get('metadata'):
                print(f"   Metadata: {result['metadata']}")
            
            print(f"   Text: {result['text'][:150]}...")
            print()
        print("="*80)


def main():
    """Main search interface"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Search the OpenSearch vector database using semantic similarity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive search mode
  ./search.py
  
  # Search with specific query
  ./search.py "artificial intelligence"
  
  # Search in custom index
  ./search.py --index observability "kubernetes pods"
  
  # Search with more results
  ./search.py --k 10 "docker configuration"
        """
    )
    
    parser.add_argument(
        'query',
        type=str,
        nargs='*',
        help='Search query (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--index',
        type=str,
        default='documents',
        help='Name of the OpenSearch index to search (default: documents)'
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
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Initialize search engine
    searcher = VectorSearch(
        opensearch_host=args.host,
        opensearch_port=args.port,
        model_name=args.model
    )
    
    # Check if query provided as command line argument
    if args.query:
        query = ' '.join(args.query)
        results = searcher.search(query, index_name=args.index, k=args.k)
        searcher.display_results(results)
    else:
        # Interactive mode
        print("="*80)
        print("🔍 Vector Database Search")
        print("="*80)
        print(f"Index: {args.index}")
        print(f"Results per query: {args.k}")
        print("\nType your search query and press Enter.")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                query = input("🔎 Search: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not query:
                    print("⚠️  Please enter a search query.\n")
                    continue
                
                print()
                results = searcher.search(query, index_name=args.index, k=args.k)
                searcher.display_results(results)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
