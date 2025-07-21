# app/ingestion/multimodal_processor.py 

import asyncio
import base64
from typing import Dict, Any, Optional
import aiohttp
from PIL import Image
import io

class MultimodalProcessor:
    """
    Handles multimodal content processing using Gemini 1.5 Flash
    """
    
    def __init__(self, gemini_api_key: str, config):
        self.api_key = gemini_api_key
        self.config = config
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.logger = config.get_logger(__name__)
    
    async def process_image_element(self, image_data: bytes, context: str = "") -> str:
        """
        Process image/chart/diagram using Gemini Vision
        
        Args:
            image_data: Raw image bytes
            context: Surrounding text context for better understanding
            
        Returns:
            Detailed text description of the visual content
        """
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Construct prompt for comprehensive analysis
            prompt = self._build_image_analysis_prompt(context)
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024,
                }
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        description = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        self.logger.info("Successfully processed visual content with Gemini")
                        return description
                    else:
                        error_msg = data.get('error', {}).get('message', 'Unknown error')
                        self.logger.warning(f"Gemini vision processing failed: {error_msg}")
                        return f"[Visual content - description unavailable: {error_msg}]"
                        
        except Exception as e:
            self.logger.error(f"Error processing image with Gemini: {str(e)}")
            return "[Visual content - description unavailable due to processing error]"
    
    async def process_table_element(self, table_html: str, context: str = "") -> str:
        """
        Process complex table using Gemini for semantic understanding
        
        Args:
            table_html: HTML representation of the table
            context: Surrounding context
            
        Returns:
            Structured text description of table content and insights
        """
        try:
            prompt = f"""
            Analyze this table and provide a comprehensive summary:

            Context: {context}
            
            Table HTML: {table_html}
            
            Please provide:
            1. A clear description of what the table shows
            2. Key data points and trends
            3. Column headers and their significance
            4. Notable patterns or insights
            5. A structured summary that would be useful for Q&A
            
            Format your response as clear, searchable text that captures the table's semantic meaning.
            """
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024,
                }
            }
            
            url = f"{self.base_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    data = await response.json()
                    
                    if response.status == 200:
                        analysis = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        return f"[TABLE ANALYSIS]\n{analysis}\n[ORIGINAL TABLE]\n{table_html}"
                    else:
                        return f"[TABLE]\n{table_html}"
                        
        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return f"[TABLE]\n{table_html}"
    
    def _build_image_analysis_prompt(self, context: str) -> str:
        """Build comprehensive prompt for image analysis"""
        return f"""
        Analyze this image/chart/diagram in detail. Provide a comprehensive description that includes:

        1. **Content Type**: What type of visual is this? (chart, diagram, photo, schematic, etc.)
        2. **Main Elements**: Key components, text, labels, symbols visible
        3. **Data/Information**: Any numerical data, trends, relationships shown
        4. **Structure**: Layout, organization, flow of information
        5. **Context Integration**: How this relates to: {context}
        6. **Searchable Summary**: A concise summary that would help someone find this content later

        Make your description detailed enough that someone could understand the image's content and significance without seeing it.
        Focus on factual, objective description rather than interpretation.
        """
