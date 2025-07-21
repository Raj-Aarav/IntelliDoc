# chunk_processor.py
from app.config import PipelineConfig
from app.ingestion.document_parser import DocumentElement
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    text: str
    metadata: Dict[str, Any]
    chunk_index: int
    start_char: int
    end_char: int

class ChunkProcessor:
    """
    Smart text chunking processor that implements recursive character splitting
    with logical separators and metadata preservation.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def process_elements(self, elements: List[DocumentElement]) -> List[DocumentChunk]:
        """
        Process document elements into chunks with metadata.
        
        Args:
            elements: List of DocumentElement objects from parser
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        current_section = None
        
        for element in elements:
            try:
                # Extract section titles for context
                if element.element_type == "Title":
                    current_section = element.text
                
                # Process element text into chunks
                element_chunks = self._split_text_recursive(
                    text=element.text,
                    base_metadata=element.metadata,
                    page_number=element.page_number,
                    section_title=current_section,
                    element_type=element.element_type
                )
                
                chunks.extend(element_chunks)
                
            except Exception as e:
                self.logger.warning(f"Error processing element: {str(e)}")
                continue
        
        # Assign chunk indices
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(elements)} elements")
        return chunks
    
    def _split_text_recursive(self, 
                             text: str, 
                             base_metadata: Dict[str, Any],
                             page_number: Optional[int],
                             section_title: Optional[str],
                             element_type: str) -> List[DocumentChunk]:
        """
        Recursively split text using logical separators.
        
        Args:
            text: Text to split
            base_metadata: Base metadata to inherit
            page_number: Page number from source
            section_title: Current section title for context
            element_type: Type of the source element
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        
        # Skip empty or very short text
        if not text or len(text.strip()) < 10:
            return chunks
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.config.chunking.chunk_size:
            chunk = self._create_chunk(
                text=text,
                base_metadata=base_metadata,
                page_number=page_number,
                section_title=section_title,
                element_type=element_type,
                start_char=0,
                end_char=len(text)
            )
            return [chunk]
        
        # Try to split using separators
        for separator in self.config.chunking.separators:
            if separator in text:
                splits = text.split(separator)
                return self._merge_splits(
                    splits=splits,
                    separator=separator,
                    base_metadata=base_metadata,
                    page_number=page_number,
                    section_title=section_title,
                    element_type=element_type
                )
        
        # If no separator works, split by character count
        return self._split_by_length(
            text=text,
            base_metadata=base_metadata,
            page_number=page_number,
            section_title=section_title,
            element_type=element_type
        )
    
    def _merge_splits(self, 
                     splits: List[str],
                     separator: str,
                     base_metadata: Dict[str, Any],
                     page_number: Optional[int],
                     section_title: Optional[str],
                     element_type: str) -> List[DocumentChunk]:
        """Merge split text pieces into chunks with overlap"""
        chunks = []
        current_chunk = ""
        start_char = 0
        
        for i, split in enumerate(splits):
            # Add separator back (except for empty separator)
            if separator and i > 0:
                potential_chunk = current_chunk + separator + split
            else:
                potential_chunk = current_chunk + split
            
            # Check if adding this split exceeds chunk size
            if len(potential_chunk) > self.config.chunking.chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    text=current_chunk.strip(),
                    base_metadata=base_metadata,
                    page_number=page_number,
                    section_title=section_title,
                    element_type=element_type,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.config.chunking.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + separator + split if separator else current_chunk[overlap_start:] + split
                start_char = start_char + overlap_start
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                text=current_chunk.strip(),
                base_metadata=base_metadata,
                page_number=page_number,
                section_title=section_title,
                element_type=element_type,
                start_char=start_char,
                end_char=start_char + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_length(self,
                        text: str,
                        base_metadata: Dict[str, Any],
                        page_number: Optional[int],
                        section_title: Optional[str],
                        element_type: str) -> List[DocumentChunk]:
        """Split text by character length with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunking.chunk_size
            
            # Try to end at word boundary
            if end < len(text):
                # Look for space within last 50 characters
                space_pos = text.rfind(' ', end - 50, end)
                if space_pos > start:
                    end = space_pos + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = self._create_chunk(
                    text=chunk_text,
                    base_metadata=base_metadata,
                    page_number=page_number,
                    section_title=section_title,
                    element_type=element_type,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.config.chunking.chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def _create_chunk(self,
                     text: str,
                     base_metadata: Dict[str, Any],
                     page_number: Optional[int],
                     section_title: Optional[str],
                     element_type: str,
                     start_char: int,
                     end_char: int) -> DocumentChunk:
        """Create a DocumentChunk with complete metadata"""
        
        # Build comprehensive metadata
        metadata = {
            "source_document": base_metadata.get("source_document"),
            "page_number": page_number,
            "chunk_index": 0,  # will be overwritten globally
            "document_type": base_metadata.get("document_type"),
            "section_title": section_title,
            "element_type": element_type,
            "created_timestamp": datetime.now().isoformat()
        }
        
        # Here, we strip the nested metadata (CRITICAL FIX)
        metadata = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        
        return DocumentChunk(
            text=text,
            metadata=metadata,
            chunk_index=0,  # Will be set later
            start_char=start_char,
            end_char=end_char
        )
