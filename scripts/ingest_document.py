# scripts/ingest_document.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import asyncio
from pathlib import Path
from app.ingestion.pipeline import DocumentIngestionPipeline

async def main():
    # Initialize pipeline with config
    pipeline = DocumentIngestionPipeline("config.json")
    
    # Process documents - Updated path
    document_paths = [
        "data/documents/research_paper_1.pdf"  # Updated to match new structure
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
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
