# reranker.py - Complete updated version with model fallbacks

import logging
from typing import List, Dict, Any
import cohere

class ReRanker:
    """
    Reranks retrieved chunks using Cohere's rerank API with multiple model fallbacks.
    """
    
    def __init__(self, cohere_api_key: str, config):
        self.cohere_client = cohere.Client(cohere_api_key)
        self.config = config
        self.logger = config.get_logger(__name__)
        
        # List of models to try in order of preference
        self.models_to_try = [
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-english-v2.0",
            "rerank-multilingual-v2.0"
        ]

    async def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """Rerank candidates using Cohere rerank API with fallback models."""
        try:
            if not candidates:
                return []
                
            # Extract text from candidates
            texts = [candidate['text'] for candidate in candidates]
            
            # Try different models until one works
            response = None
            working_model = None
            
            for model in self.models_to_try:
                try:
                    self.logger.info(f"Trying rerank model: {model}")
                    response = self.cohere_client.rerank(
                        query=query,
                        documents=texts,
                        model=model
                    )
                    working_model = model
                    self.logger.info(f"Successfully using rerank model: {model}")
                    break
                except Exception as model_error:
                    self.logger.warning(f"Model {model} failed: {str(model_error)}")
                    continue
            
            if response is None:
                raise Exception("All rerank models failed")
            
            # Build reranked results and manually limit to top_k
            reranked = []
            for i, result in enumerate(response.results):
                if i >= top_k:  # Manually limit results
                    break
                original_candidate = candidates[result.index].copy()
                original_candidate['rerank_score'] = result.relevance_score
                reranked.append(original_candidate)
            
            self.logger.info(f"Successfully reranked {len(candidates)} to {len(reranked)} chunks using {working_model}")
            return reranked
            
        except Exception as e:
            self.logger.warning(f"All reranking failed: {str(e)}, using enhanced fallback order")
            # Enhanced fallback: sort by text length to prefer more informative chunks
            sorted_candidates = sorted(candidates, key=lambda x: len(x['text']), reverse=True)
            return sorted_candidates[:top_k]

    def get_available_models(self) -> List[str]:
        """Test which rerank models are available for this API key."""
        available = []
        for model in self.models_to_try:
            try:
                # Test with minimal query
                self.cohere_client.rerank(
                    query="test",
                    documents=["test document"],
                    model=model
                )
                available.append(model)
            except Exception:
                continue
        return available
