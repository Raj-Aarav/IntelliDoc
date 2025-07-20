# vector_store.py - Fix the embedding API call

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import uuid

# Third-party imports
import chromadb
from chromadb.config import Settings
import cohere

class VectorStore:
    """
    Vector database operations for storing and retrieving document embeddings.
    Updated for Cohere API v4+ compatibility.
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding client
        self.cohere_client = cohere.Client(config.embedding.api_key)
        
        # Initialize ChromaDB with telemetry disabled
        self.chroma_client = chromadb.PersistentClient(
            path=config.vector_db.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.vector_db.collection_name,
            metadata={"hnsw:space": config.vector_db.distance_metric}
        )
        
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using updated Cohere API."""
        try:
            # Use the correct Cohere v4+ API format (removed embedding_types)
            response = self.cohere_client.embed(
                texts=texts,
                model=self.config.embedding.model_name,
                input_type=self.config.embedding.input_type
            )
            
            # Access embeddings directly (no .float attribute in v4+)
            return response.embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            # Try fallback approach
            try:
                response = self.cohere_client.embed(
                    texts=texts,
                    model="embed-english-v3.0",  # Fallback model
                    input_type="search_document"
                )
                return response.embeddings
            except Exception as fallback_error:
                self.logger.error(f"Fallback embedding failed: {str(fallback_error)}")
                raise

    async def query_similar_chunks(self, 
                                 query_text: str, 
                                 top_k: int = 10,
                                 filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query for similar chunks with corrected API usage."""
        try:
            # Generate embedding for query
            query_response = self.cohere_client.embed(
                texts=[query_text],
                model=self.config.embedding.model_name,
                input_type="search_query"  # Different for queries
            )
            
            query_embedding = query_response.embeddings[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Error querying similar chunks: {str(e)}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            return {
                'total_chunks': count,
                'collection_name': self.config.vector_db.collection_name,
                'distance_metric': self.config.vector_db.distance_metric,
                'embedding_model': self.config.embedding.model_name
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    async def delete_by_source(self, source_document: str) -> int:
        """Delete all chunks from a specific source document"""
        try:
            # Get all chunks from source
            results = self.collection.get(
                where={"source_document": source_document}
            )
            
            if results['ids']:
                # Delete chunks
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error deleting chunks from {source_document}: {str(e)}")
            raise
