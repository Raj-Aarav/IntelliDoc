# rag_pipeline.py - Enhanced version with better error handling and performance

import asyncio
import time
from typing import Dict, Any, List
from query_processor import QueryProcessor
from reranker import ReRanker
from prompt_builder import PromptBuilder

class RAGPipeline:
    """
    Enhanced RAG pipeline orchestrator with improved error handling and performance monitoring.
    """
    
    def __init__(self, 
                 query_processor: QueryProcessor,
                 reranker: ReRanker,
                 prompt_builder: PromptBuilder,
                 llm_client,
                 cache: Dict[str, Any],
                 config):
        """Initialize enhanced RAG pipeline."""
        self.query_processor = query_processor
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.llm_client = llm_client
        self.cache = cache
        self.config = config
        self.logger = config.get_logger(__name__)
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0
        self.cache_hits = 0

    async def process_query(self, query: str, enable_reranking: bool = True) -> str:
        """Enhanced query processing with detailed performance tracking."""
        start_time = time.time()
        self.query_count += 1
        
        try:
            # Cache check with improved key generation
            cache_key = f"rag_v2_{hash(query.lower().strip())}"
            if cache_key in self.cache:
                self.cache_hits += 1
                self.logger.info(f"Cache hit for query: {query[:50]}...")
                return self.cache[cache_key]

            self.logger.info(f"Processing query {self.query_count}: {query[:100]}...")

            # Step 1: Enhanced retrieval
            self.logger.info("Step 1: Enhanced retrieval with query expansion...")
            candidates = await self.query_processor.retrieve_candidates(query, top_k=25)
            
            if not candidates:
                return "I could not find any relevant information to answer your question."

            # Filter candidates by relevance
            filtered_candidates = await self.query_processor.filter_by_relevance(candidates, query)
            if filtered_candidates:
                candidates = filtered_candidates

            self.logger.info(f"Retrieved and filtered to {len(candidates)} candidates")

            # Step 2: Enhanced reranking with fallback
            reranked_chunks = candidates[:6]  # Default fallback
            
            if enable_reranking:
                self.logger.info("Step 2: Enhanced reranking...")
                try:
                    reranked_chunks = await self.reranker.rerank(query, candidates, top_k=6)
                    self.logger.info(f"Successfully reranked to {len(reranked_chunks)} chunks")
                except Exception as rerank_error:
                    self.logger.warning(f"Reranking failed, using enhanced fallback: {str(rerank_error)}")
                    # Enhanced fallback: combine length and relevance
                    scored_candidates = []
                    for candidate in candidates[:12]:  # Consider top 12
                        score = len(candidate['text']) * 0.1 + candidate.get('term_overlap', 0) * 10
                        candidate['fallback_score'] = score
                        scored_candidates.append(candidate)
                    
                    reranked_chunks = sorted(scored_candidates, key=lambda x: x['fallback_score'], reverse=True)[:6]

            # Step 3: Enhanced prompt building
            self.logger.info("Step 3: Building enhanced prompt...")
            
            # Validate context quality
            context_quality = self.prompt_builder.validate_context_quality(reranked_chunks, query)
            self.logger.info(f"Context quality score: {context_quality['quality_score']}/100")
            
            if context_quality['issues']:
                self.logger.warning(f"Context issues: {context_quality['issues']}")
            
            prompt = self.prompt_builder.build_prompt(query, reranked_chunks)
            
            # Log prompt size for debugging
            self.logger.info(f"Generated prompt with {len(prompt)} characters")

            # Step 4: Enhanced LLM generation
            self.logger.info("Step 4: Generating enhanced answer...")
            answer = await self.llm_client.generate(prompt)
            
            # Step 5: Answer post-processing and validation
            processed_answer = await self._post_process_answer(answer, reranked_chunks)
            
            # Performance logging
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            avg_time = self.total_processing_time / self.query_count
            
            self.logger.info(f"Query processed successfully in {processing_time:.2f}s (avg: {avg_time:.2f}s)")
            
            # Cache the result
            self.cache[cache_key] = processed_answer
            
            return processed_answer
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error processing query in {processing_time:.2f}s: {str(e)}")
            return f"I encountered an error while processing your question. Please try rephrasing your query."

    async def _post_process_answer(self, answer: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Post-process and validate the generated answer."""
        # Check for citation presence
        has_citations = '[' in answer and ']' in answer and 'page' in answer.lower()
        
        # If no citations but not a "not found" response, add context note
        if not has_citations and "could not find" not in answer.lower():
            available_sources = set()
            for chunk in context_chunks:
                metadata = chunk.get('metadata', {})
                source = metadata.get('source_document', 'unknown')
                page = metadata.get('page_number', 'unknown')
                available_sources.add(f"{source}, page {page}")
            
            if available_sources:
                sources_note = f"\n\n*Sources consulted: {'; '.join(list(available_sources)[:3])}*"
                answer += sources_note
        
        return answer

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced pipeline performance statistics."""
        try:
            # Get vector store stats
            vector_stats = await self.query_processor.vector_store.get_collection_stats()
            
            cache_hit_rate = (self.cache_hits / max(self.query_count, 1)) * 100
            avg_processing_time = self.total_processing_time / max(self.query_count, 1)
            
            return {
                "queries_processed": self.query_count,
                "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                "average_processing_time": f"{avg_processing_time:.2f}s",
                "cache_size": len(self.cache),
                "vector_store_stats": vector_stats,
                "config": {
                    "embedding_model": self.config.embedding.model_name,
                    "llm_backend": self.config.phase2.llm.backend,
                    "reranker": self.config.phase2.reranker_type
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {str(e)}")
            return {"error": str(e)}
