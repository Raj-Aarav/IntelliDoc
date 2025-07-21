# scripts/interactive_rag_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

import asyncio
import time
from datetime import datetime

from app.ingestion.vector_store import VectorStore
from app.query.query_processor import QueryProcessor
from app.query.reranker import ReRanker
from app.query.prompt_builder import PromptBuilder
from app.query.rag_pipeline import RAGPipeline
from app.config import PipelineConfig  # Fixed import
from app.llm_clients.gemini_client import GeminiLLMClient
from app.llm_clients.groq_client import GroqLLMClient
import cohere

class InteractiveRAGApplication:
    """Complete Interactive RAG Application"""
    
    def __init__(self, config_path="config.json"):
        self.config = PipelineConfig.from_json(config_path)
        self.rag_pipeline = None
        self.session_stats = {
            "queries_processed": 0,
            "start_time": datetime.now(),
            "total_processing_time": 0
        }
    
    async def initialize_system(self):
        """Initialize all system components"""
        print("🚀 Initializing Interactive RAG System...")
        print("=" * 60)
        
        try:
            # Test Cohere connection
            cohere_client = cohere.Client(self.config.embedding.api_key)
            test_embed = cohere_client.embed(
                texts=["test"], 
                model=self.config.embedding.model_name,
                input_type=self.config.embedding.input_type
            )
            print("✅ Cohere embedding API: Connected")
            
            # Initialize components
            vector_store = VectorStore(self.config)
            
            # Check vector store has data
            stats = await vector_store.get_collection_stats()
            if stats.get('total_chunks', 0) == 0:
                print("❌ No documents found in vector store!")
                print("   Please run: python scripts/ingest_document.py (to ingest documents first)")
                return False
            
            print(f"📊 Loaded {stats.get('total_chunks')} document chunks")
            
            query_processor = QueryProcessor(vector_store, cohere_client, self.config)
            reranker = ReRanker(self.config.phase2.cohere_api_key, self.config)
            prompt_builder = PromptBuilder()
            
            # Initialize LLM
            if self.config.phase2.llm.backend == "gemini":
                llm_client = GeminiLLMClient(self.config.phase2.llm.gemini_api_key, self.config)
            else:
                llm_client = GroqLLMClient(
                    self.config.phase2.llm.groq_api_key,
                    self.config.phase2.llm.groq_model,
                    self.config
                )
            
            # Create RAG pipeline
            self.rag_pipeline = RAGPipeline(
                query_processor=query_processor,
                reranker=reranker,
                prompt_builder=prompt_builder,
                llm_client=llm_client,
                cache={},
                config=self.config
            )
            
            print("✅ RAG Pipeline initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def process_user_query(self, query: str):
        """Process a single user query"""
        start_time = time.time()
        
        try:
            answer = await self.rag_pipeline.process_query(query)
            processing_time = time.time() - start_time
            
            # Update stats
            self.session_stats["queries_processed"] += 1
            self.session_stats["total_processing_time"] += processing_time
            
            return answer, processing_time
            
        except Exception as e:
            return f"Error processing your question: {str(e)}", 0
    
    def display_session_stats(self):
        """Display session statistics"""
        duration = (datetime.now() - self.session_stats["start_time"]).total_seconds()
        avg_time = (self.session_stats["total_processing_time"] / 
                   max(self.session_stats["queries_processed"], 1))
        
        print(f"\n📊 Session Statistics:")
        print(f"   • Total queries: {self.session_stats['queries_processed']}")
        print(f"   • Session duration: {duration:.1f} seconds")
        print(f"   • Average response time: {avg_time:.2f} seconds")
        print(f"   • Total processing time: {self.session_stats['total_processing_time']:.2f} seconds")
    
    async def run_interactive_session(self):
        """Main interactive loop"""
        if not await self.initialize_system():
            return
        
        print("\n" + "="*60)
        print("🎯 INTERACTIVE RAG SYSTEM READY")
        print("="*60)
        print("💡 Ask questions about your documents!")
        print("💡 Type 'help' for commands, 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                # Get user input
                print(f"\n[Query #{self.session_stats['queries_processed'] + 1}]")
                user_query = input("🤔 Your question: ").strip()
                
                # Handle special commands
                if user_query.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_query.lower() == 'help':
                    self.show_help()
                    continue
                elif user_query.lower() == 'stats':
                    self.display_session_stats()
                    continue
                elif not user_query:
                    continue
                
                # Process the query
                print("🔍 Processing your question...")
                answer, processing_time = await self.process_user_query(user_query)
                
                # Display results
                print(f"\n🤖 Answer (processed in {processing_time:.2f}s):")
                print("-" * 50)
                print(answer)
                print("-" * 50)
                
                # Ask for feedback (optional)
                feedback = input("\n👍 Was this helpful? (y/n/skip): ").strip().lower()
                if feedback == 'y':
                    print("😊 Great! Thank you for the feedback.")
                elif feedback == 'n':
                    print("😔 Sorry about that. Try rephrasing your question for better results.")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Show final statistics
        print("\n" + "="*60)
        print("👋 Thank you for using the Interactive RAG System!")
        self.display_session_stats()
        print("="*60)
    
    def show_help(self):
        """Display help information"""
        print("\n📖 Available Commands:")
        print("   • Ask any question about your documents")
        print("   • 'help' - Show this help message")
        print("   • 'stats' - Show session statistics")
        print("   • 'quit' or 'exit' - End the session")
        print("\n💡 Tips:")
        print("   • Be specific in your questions")
        print("   • Ask about information that should be in your documents")
        print("   • Check the citations in square brackets [document.pdf, page X]")

async def main():
    """Main application entry point"""
    app = InteractiveRAGApplication()
    await app.run_interactive_session()

if __name__ == "__main__":
    # Handle Windows event loop issues
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())
