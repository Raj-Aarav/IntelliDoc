# vector_store.py - Complete fix with missing methods

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

    # CRITICAL FIX: Add the missing embed_and_store_chunks method
    async def embed_and_store_chunks(self, chunks) -> Dict[str, Any]:
        """
        Embed chunks and store them in vector database.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            Dictionary with processing statistics
        """
        if not chunks:
            return {"total_chunks": 0, "successful": 0, "failed": 0}
        
        start_time = datetime.now()
        successful = 0
        failed = 0

        batch_size = self.config.embedding.batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                embeddings = await self._generate_embeddings([chunk.text for chunk in batch])
                await self._store_batch(batch, embeddings)
                successful += len(batch)
                self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed += len(batch)
                continue

        processing_time = (datetime.now() - start_time).total_seconds()
        return {
            "total_chunks": len(chunks),
            "successful": successful,
            "failed": failed,
            "processing_time_seconds": processing_time,
            "chunks_per_second": successful / processing_time if processing_time > 0 else 0
        }

    async def _store_batch(self, chunks, embeddings: List[List[float]]) -> None:
        """Store a batch of chunks with their embeddings"""
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = []
        
        # Clean metadata for ChromaDB compatibility
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            # Remove any nested objects that ChromaDB can't handle
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    clean_metadata[key] = value
                else:
                    # Convert other types to string
                    clean_metadata[key] = str(value)
            
            clean_metadata['chunk_index'] = chunk.chunk_index
            clean_metadata['embedding_timestamp'] = datetime.now().isoformat()
            metadatas.append(clean_metadata)
        
        try:
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        except Exception as e:
            self.logger.error(f"Error storing batch in ChromaDB: {str(e)}")
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
