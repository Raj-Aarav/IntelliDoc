# main.py
import asyncio
from pathlib import Path
from pipeline import DocumentIngestionPipeline

async def main():
    # Initialize pipeline with config
    pipeline = DocumentIngestionPipeline("config.json")
    
    # Process documents
    document_paths = [
        "documents/research_paper.pdf"
    ]
    
    try:
        # Process all documents
        results = await pipeline.process_documents(document_paths)
        print(f"Processing complete: {results}")
        
        # Get pipeline statistics
        stats = await pipeline.get_pipeline_stats()
        print(f"Pipeline stats: {stats}")
        
        # Test search functionality
        search_results = await pipeline.search_documents("Load Balancing", top_k=5)
        print(f"Search results: {len(search_results)} chunks found")
        
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"Source: {result['metadata']['source_document']}")
            print(f"Page: {result['metadata']['page_number']}")
            print(f"Text: {result['text'][:200]}...")
            
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
