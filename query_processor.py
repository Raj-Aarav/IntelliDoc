# query_processor.py - Enhanced version with better query processing

import asyncio
import logging
from typing import List, Dict, Any, Set
from vector_store import VectorStore
import cohere
import re

class QueryProcessor:
    """
    Handles enhanced query preprocessing and retrieval from vector store.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_client: cohere.Client, config):
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.config = config
        self.logger = config.get_logger(__name__)

    async def preprocess_query(self, query: str) -> str:
        """Enhanced query cleaning and normalization."""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to lowercase for consistency
        cleaned = cleaned.lower()
        
        # Remove common stop words that don't add semantic value
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'how', 'why'}
        words = cleaned.split()
        filtered_words = [w for w in words if w not in stop_words or len(words) <= 3]
        
        return ' '.join(filtered_words)

    async def expand_query(self, query: str) -> List[str]:
        """Generate query variations to improve recall."""
        base_query = await self.preprocess_query(query)
        expanded_queries = [base_query]
        
        # Add synonym variations for common terms
        synonyms = {
            'algorithm': ['method', 'approach', 'technique'],
            'load balancing': ['load distribution', 'workload distribution'],
            'dynamic': ['adaptive', 'flexible'],
            'static': ['fixed', 'predetermined'],
            'method': ['technique', 'approach', 'strategy']
        }
        
        for term, alternatives in synonyms.items():
            if term in base_query:
                for alt in alternatives:
                    expanded_queries.append(base_query.replace(term, alt))
        
        # Limit to avoid too many queries
        return expanded_queries[:3]

    async def retrieve_candidates(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Enhanced retrieval with query expansion and deduplication."""
        try:
            # Get expanded queries
            queries = await self.expand_query(query)
            self.logger.info(f"Using {len(queries)} query variations")
            
            all_results = []
            seen_texts: Set[str] = set()
            
            for q in queries:
                # Search vector store
                results = await self.vector_store.query_similar_chunks(
                    query_text=q,
                    top_k=top_k
                )
                
                # Deduplicate by text content
                for result in results:
                    text_key = result['text'][:200]  # Use first 200 chars as key
                    if text_key not in seen_texts:
                        seen_texts.add(text_key)
                        all_results.append(result)
            
            # Sort by distance/similarity score and limit results
            if all_results and 'distance' in all_results[0]:
                all_results.sort(key=lambda x: x['distance'])
            
            final_results = all_results[:top_k]
            
            self.logger.info(f"Retrieved {len(final_results)} unique candidates from {len(all_results)} total results")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced retrieval: {str(e)}")
            return []

    async def filter_by_relevance(self, candidates: List[Dict[str, Any]], query: str, min_length: int = 100) -> List[Dict[str, Any]]:
        """Filter candidates by basic relevance criteria."""
        filtered = []
        query_terms = set(query.lower().split())
        
        for candidate in candidates:
            text = candidate['text'].lower()
            
            # Filter out very short chunks
            if len(candidate['text']) < min_length:
                continue
            
            # Check for query term overlap
            text_terms = set(text.split())
            overlap = len(query_terms & text_terms)
            
            if overlap > 0:  # At least one term matches
                candidate['term_overlap'] = overlap
                filtered.append(candidate)
        
        # Sort by term overlap
        filtered.sort(key=lambda x: x.get('term_overlap', 0), reverse=True)
        
        self.logger.info(f"Filtered {len(candidates)} to {len(filtered)} relevant candidates")
        return filtered
