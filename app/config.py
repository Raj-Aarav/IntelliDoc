# # config.py

from dotenv import load_dotenv
load_dotenv()
import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model_name: str = "embed-english-v3.0"
    api_key: Optional[str] = None
    batch_size: int = 96
    input_type: str = "search_document"
    
@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

@dataclass
class VectorDBConfig:
    """Configuration for vector database"""
    db_type: str = "chromadb"
    collection_name: str = "documents"
    persist_directory: str = "./data/chroma_db"  # Updated path
    distance_metric: str = "cosine"

@dataclass
class LLMConfig:
    """Configuration for LLM backends"""
    backend: str = "gemini"  # "gemini" or "groq"
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: str = "llama3-70b-8192"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings"""
    initial_top_k: int = 20
    rerank_top_k: int = 4
    max_context_chars: int = 4000

@dataclass
class Phase2Config:
    """Configuration for Phase 2 components"""
    reranker_type: str = "cohere"
    cohere_api_key: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache_size: int = 256

    def get_logger(self, name: str):
        """Get configured logger instance"""
        logger = logging.getLogger(name)
        if not logger.handlers:
            # Set up logging to logs directory
            log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "pipeline.log")
            
            handler = logging.FileHandler(log_path, mode='a')
            console_handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        return logger

@dataclass
class PipelineConfig:
    """Main pipeline configuration that includes both Phase 1 and Phase 2"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    supported_formats: List[str] = field(default_factory=lambda: ['.pdf', '.docx', '.pptx', '.txt'])
    max_file_size_mb: int = 100
    
    @classmethod
    def from_json(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file with .env override"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Build nested configurations with .env overrides
        embedding_data = config_data.get('embedding', {})
        embedding_config = EmbeddingConfig(
            model_name=embedding_data.get('model_name', 'embed-english-v3.0'),
            api_key=os.getenv('COHERE_API_KEY') or embedding_data.get('api_key'),  # .env override
            batch_size=embedding_data.get('batch_size', 96),
            input_type=embedding_data.get('input_type', 'search_document')
        )
        
        chunking_config = ChunkingConfig(**config_data.get('chunking', {}))
        vector_db_config = VectorDBConfig(**config_data.get('vector_db', {}))
        
        # Build Phase 2 configuration with .env overrides
        phase2_data = config_data.get('phase2', {})
        llm_data = phase2_data.get('llm', {})
        retrieval_data = phase2_data.get('retrieval', {})
        
        llm_config = LLMConfig(
            backend=llm_data.get('backend', 'gemini'),
            gemini_api_key=os.getenv('GEMINI_API_KEY') or llm_data.get('gemini_api_key'),  # .env override
            groq_api_key=os.getenv('GROQ_API_KEY') or llm_data.get('groq_api_key'),  # .env override
            groq_model=llm_data.get('groq_model', 'llama3-70b-8192')
        )
        
        retrieval_config = RetrievalConfig(**retrieval_data)
        phase2_config = Phase2Config(
            reranker_type=phase2_data.get('reranker_type', 'cohere'),
            cohere_api_key=os.getenv('COHERE_API_KEY') or phase2_data.get('cohere_api_key', ''),  # .env override
            llm=llm_config,
            retrieval=retrieval_config,
            cache_size=phase2_data.get('cache_size', 256)
        )
        
        return cls(
            embedding=embedding_config,
            chunking=chunking_config,
            vector_db=vector_db_config,
            phase2=phase2_config,
            supported_formats=config_data.get('supported_formats', ['.pdf', '.docx', '.pptx', '.txt']),
            max_file_size_mb=config_data.get('max_file_size_mb', 100)
        )

    def get_logger(self, name: str):
        """Get configured logger instance"""
        return self.phase2.get_logger(name)
