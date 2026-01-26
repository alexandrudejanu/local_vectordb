#!/usr/bin/env python3
"""
Search the vector database using semantic similarity
"""

from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
import sys
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
                'metadata': hit['_source'].get('metadata', '')
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
            print(f"   Category: {result['metadata']}")
            print(f"   Text: {result['text'][:150]}...")
            print()
        print("="*80)


def main():
    """Main search interface"""
    
    # Initialize search engine
    searcher = VectorSearch(
        opensearch_host="localhost",
        opensearch_port=9200,
        model_name="all-MiniLM-L6-v2"
    )
    
    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        results = searcher.search(query, index_name="documents", k=5)
        searcher.display_results(results)
    else:
        # Interactive mode
        print("="*80)
        print("🔍 Vector Database Search")
        print("="*80)
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
                results = searcher.search(query, index_name="documents", k=5)
                searcher.display_results(results)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
