# prompt_builder.py - Enhanced version with better context management

from typing import List, Dict, Any

class PromptBuilder:
    """
    Builds prompts for LLM with enhanced context chunks and user questions.
    """
    
    SYSTEM_PROMPT = """You are an expert Q&A assistant that answers questions based ONLY on the provided context. Follow these rules strictly:

1. Synthesize a comprehensive answer to the user's question using ALL relevant information from the context below.
2. For every piece of information you use, you MUST cite the source document and page number in square brackets, like [document_name.pdf, page X].
3. If multiple sources discuss the same topic, cite all relevant sources.
4. If the answer cannot be found in the provided context, respond with: "I could not find an answer to your question in the provided documents."
5. Do not use any information outside the provided context.

Context:
{context}

User Question: {user_question}

Please provide a detailed answer with proper citations."""

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
        """Build enhanced context string from chunks with better formatting."""
        context_parts = []
        current_length = 0
        chunk_count = 0
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source_doc = metadata.get('source_document', 'unknown')
            page_num = metadata.get('page_number', 'unknown')
            section_title = metadata.get('section_title', '')
            
            # Enhanced chunk formatting with section context
            chunk_text = chunk['text']
            if len(chunk_text) > 1200:  # Increased chunk size limit
                chunk_text = chunk_text[:1200] + "..."
            
            # Build context block with enhanced metadata
            context_header = f"--- Source {chunk_count + 1} ---"
            if section_title:
                context_header += f" [Section: {section_title}]"
            
            context_block = f"{context_header}\n{chunk_text}\n[Citation: {source_doc}, page {page_num}]\n\n"
            
            if current_length + len(context_block) > max_chars:
                break
                
            context_parts.append(context_block)
            current_length += len(context_block)
            chunk_count += 1
        
        context_summary = f"=== CONTEXT SUMMARY ===\nTotal sources provided: {chunk_count}\nSources span: {source_doc}\n\n"
        return context_summary + "".join(context_parts)

    @classmethod
    def build_prompt(cls, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build complete prompt for LLM with enhanced context."""
        context = cls.build_context(context_chunks)
        return cls.SYSTEM_PROMPT.format(context=context, user_question=question)

    @staticmethod
    def validate_context_quality(chunks: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Validate the quality and relevance of context chunks."""
        if not chunks:
            return {"quality_score": 0, "issues": ["No chunks provided"]}
        
        issues = []
        quality_score = 100
        
        # Check for missing metadata
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            if not metadata.get('source_document'):
                issues.append(f"Chunk {i} missing source document")
                quality_score -= 10
            if not metadata.get('page_number'):
                issues.append(f"Chunk {i} missing page number")
                quality_score -= 5
        
        # Check total context length
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        if total_chars < 500:
            issues.append("Context too short - may lack sufficient information")
            quality_score -= 20
        
        return {
            "quality_score": max(0, quality_score),
            "issues": issues,
            "total_chunks": len(chunks),
            "total_chars": total_chars
        }
