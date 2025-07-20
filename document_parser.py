# document_parser.py
from config import PipelineConfig
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import mimetypes

# Third-party imports
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from pathlib import Path

from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import convert_to_dict
import cohere

@dataclass
class DocumentElement:
    """Represents a parsed document element"""
    text: str
    element_type: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    coordinates: Optional[Dict[str, float]] = None

class DocumentParser:
    """
    Advanced document parser that handles multiple file types and preserves structure.
    Uses unstructured.io for layout detection and element extraction.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cohere_client = cohere.Client(config.embedding.api_key) if config.embedding.api_key else None
        
    async def parse_document(self, file_path: Union[str, Path]) -> List[DocumentElement]:
        """
        Parse a document and return structured elements.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of DocumentElement objects
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        # Validate file
        self._validate_file(file_path)
        
        try:
            # Parse document using unstructured
            elements = await self._parse_with_unstructured(file_path)
            
            # Process elements and extract structured data
            processed_elements = await self._process_elements(elements, file_path)
            
            self.logger.info(f"Successfully parsed {len(processed_elements)} elements from {file_path.name}")
            return processed_elements
            
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {str(e)}")
            raise
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate file exists and is supported format"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() not in self.config.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB")
    
    async def _parse_with_unstructured(self, file_path: Path) -> List[Any]:
        """Parse document using unstructured.io with type-aware logic"""
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == ".pdf":
                elements = partition_pdf(
                    filename=str(file_path),
                    strategy="hi_res",
                    infer_table_structure=True,
                    include_page_breaks=True,
                    include_metadata=True,
                )
            elif suffix == ".docx":
                elements = partition_docx(
                    filename=str(file_path),
                    include_metadata=True
                )
            elif suffix == ".pptx":
                elements = partition_pptx(
                    filename=str(file_path),
                    include_metadata=True,
                )
            elif suffix == ".txt":
                elements = partition_text(
                    filename=str(file_path),
                    include_metadata=True
                )
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
    
            return elements
    
        except Exception as e:
            self.logger.error(f"Unstructured parsing failed for {file_path}: {str(e)}")
            raise
    
    async def _process_elements(self, elements: List[Any], file_path: Path) -> List[DocumentElement]:
        """Process raw elements into structured DocumentElement objects"""
        processed_elements = []
        
        for element in elements:
            try:
                # Extract basic information
                text = str(element)
                element_type = element.__class__.__name__
                metadata = element.metadata.to_dict() if hasattr(element, 'metadata') else {}
                
                # Handle tables specifically
                if element_type == "Table":
                    text = self._format_table_as_markdown(element)
                
                # Handle images/charts with description generation
                elif element_type == "Image":
                    text = await self._generate_image_description(element, file_path)
                
                # Extract page number
                page_number = metadata.get('page_number')
                
                # Extract coordinates if available
                coordinates = None
                if hasattr(element, 'metadata') and element.metadata.coordinates:
                    coordinates = {
                        'x': element.metadata.coordinates.points[0][0],
                        'y': element.metadata.coordinates.points[0][1],
                        'width': element.metadata.coordinates.points[2][0] - element.metadata.coordinates.points[0][0],
                        'height': element.metadata.coordinates.points[2][1] - element.metadata.coordinates.points[0][1]
                    }
                
                # Create DocumentElement
                doc_element = DocumentElement(
                    text=text,
                    element_type=element_type,
                    metadata={
                        **metadata,
                        'source_document': file_path.name,
                        'document_type': file_path.suffix.lower().lstrip('.'),
                        'created_timestamp': datetime.now().isoformat()
                    },
                    page_number=page_number,
                    coordinates=coordinates
                )
                
                processed_elements.append(doc_element)
                
            except Exception as e:
                self.logger.warning(f"Error processing element {element}: {str(e)}")
                continue
        
        return processed_elements
    
    def _format_table_as_markdown(self, table_element) -> str:
        """Convert table element to markdown format"""
        try:
            # Check if table has structured data
            if hasattr(table_element, 'metadata') and table_element.metadata.table_as_cells:
                # Convert structured table to markdown
                cells = table_element.metadata.table_as_cells
                if not cells:
                    return str(table_element)
                
                # Group cells by row
                rows = {}
                for cell in cells:
                    row_idx = cell.get('row_index', 0)
                    if row_idx not in rows:
                        rows[row_idx] = {}
                    rows[row_idx][cell.get('col_index', 0)] = cell.get('text', '')
                
                # Build markdown table
                markdown_rows = []
                for row_idx in sorted(rows.keys()):
                    row_data = rows[row_idx]
                    row_text = "| " + " | ".join(row_data.get(col, '') for col in sorted(row_data.keys())) + " |"
                    markdown_rows.append(row_text)
                    
                    # Add header separator after first row
                    if row_idx == 0:
                        separator = "| " + " | ".join("---" for _ in row_data.keys()) + " |"
                        markdown_rows.append(separator)
                
                return "\n".join(markdown_rows)
            
            else:
                # Fallback to raw text
                return str(table_element)
                
        except Exception as e:
            self.logger.warning(f"Error formatting table as markdown: {str(e)}")
            return str(table_element)
    
    async def _generate_image_description(self, image_element, file_path: Path) -> str:
        """Generate text description for images/charts using multimodal model"""
        try:
            # For now, return a placeholder description
            # In production, you would use a multimodal model like GPT-4V or Claude 3
            description = f"[Image/Chart from {file_path.name}]: {str(image_element)}"
            
            # TODO: Implement actual image description generation
            # Example with OpenAI GPT-4V:
            # description = await self._call_vision_model(image_element)
            
            return description
            
        except Exception as e:
            self.logger.warning(f"Error generating image description: {str(e)}")
            return f"[Image/Chart from {file_path.name}]: Description unavailable"
    
    async def parse_multiple_documents(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[DocumentElement]]:
        """Parse multiple documents concurrently"""
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self.parse_document(file_path))
            tasks.append((file_path, task))
        
        results = {}
        for file_path, task in tasks:
            try:
                elements = await task
                results[str(file_path)] = elements
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {str(e)}")
                results[str(file_path)] = []
        
        return results
