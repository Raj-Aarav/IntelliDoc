# pipeline.py
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import json

# Import our classes
from app.config import PipelineConfig
from app.ingestion.document_parser import DocumentParser
from app.ingestion.chunk_processor import ChunkProcessor
from app.ingestion.vector_store import VectorStore

class DocumentIngestionPipeline:
    """
    Main pipeline orchestrator that coordinates document parsing, chunking, and vector storage.
    """
    
    def __init__(self, config: Union[PipelineConfig, str]):
        """
        Initialize the pipeline.
        
        Args:
            config: PipelineConfig object or path to config file
        """
        if isinstance(config, str):
            self.config = PipelineConfig.from_json(config)
        else:
            self.config = config
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.parser = DocumentParser(self.config)
        self.chunk_processor = ChunkProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """Configure logging for the pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler()
            ]
        )
    
    async def process_documents(self, 
                               file_paths: List[Union[str, Path]],
                               replace_existing: bool = False) -> Dict[str, Any]:
        """
        Process a list of documents through the complete pipeline.
        
        Args:
            file_paths: List of file paths to process
            replace_existing: Whether to replace existing chunks for these documents
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now()
        
        self.logger.info(f"Starting pipeline for {len(file_paths)} documents")
        
        results = {
            'total_documents': len(file_paths),
            'successful_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'processing_time_seconds': 0,
            'failed_files': []
        }
        
        try:
            # Step 1: Parse documents
            self.logger.info("Step 1: Parsing documents...")
            parsed_results = await self.parser.parse_multiple_documents(file_paths)
            
            # Step 2: Process elements into chunks
            self.logger.info("Step 2: Processing elements into chunks...")
            all_chunks = []
            
            for file_path, elements in parsed_results.items():
                if elements:
                    try:
                        # Delete existing chunks if replacing
                        if replace_existing:
                            await self.vector_store.delete_by_source(Path(file_path).name)
                        
                        # Process elements into chunks
                        chunks = await self.chunk_processor.process_elements(elements)
                        all_chunks.extend(chunks)
                        
                        results['successful_documents'] += 1
                        self.logger.info(f"Created {len(chunks)} chunks from {file_path}")
                        
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {str(e)}")
                        results['failed_documents'] += 1
                        results['failed_files'].append({'file': file_path, 'error': str(e)})
                else:
                    results['failed_documents'] += 1
                    results['failed_files'].append({'file': file_path, 'error': 'No elements parsed'})
            
            # Step 3: Embed and store chunks
            self.logger.info("Step 3: Embedding and storing chunks...")
            if all_chunks:
                storage_results = await self.vector_store.embed_and_store_chunks(all_chunks)
                
                results['total_chunks'] = storage_results['total_chunks']
                results['successful_chunks'] = storage_results['successful']
                results['failed_chunks'] = storage_results['failed']
            
            # Calculate final statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            results['processing_time_seconds'] = processing_time
            
            self.logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
            self.logger.info(f"Processed {results['successful_documents']}/{results['total_documents']} documents")
            self.logger.info(f"Created {results['successful_chunks']} chunks")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    async def process_single_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single document through the pipeline"""
        return await self.process_documents([file_path])
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics"""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            
            return {
                'pipeline_config': {
                    'embedding_model': self.config.embedding.model_name,
                    'chunk_size': self.config.chunking.chunk_size,
                    'chunk_overlap': self.config.chunking.chunk_overlap,
                    'vector_db_type': self.config.vector_db.db_type,
                    'supported_formats': self.config.supported_formats
                },
                'vector_store_stats': vector_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {str(e)}")
            return {}
    
    async def search_documents(self, 
                             query: str, 
                             top_k: int = 10,
                             filter_by_document: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_by_document: Optional document name filter
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            filter_metadata = None
            if filter_by_document:
                filter_metadata = {"source_document": filter_by_document}
            
            results = await self.vector_store.query_similar_chunks(
                query_text=query,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            raise
