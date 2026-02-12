import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from supabase import create_client, Client

class QueryLogger:
    """Log queries and metadata to Supabase for analytics."""
    
    def __init__(self):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_ANON_KEY")
        
        if not supabase_url or not supabase_key:
            print("WARNING: Supabase credentials not found. Logging disabled.")
            self.client = None
        else:
            self.client: Client = create_client(supabase_url, supabase_key)
            print(f"✓ Supabase logger initialized: {supabase_url[:30]}...")
    
    def log_query(
        self,
        session_id: str,
        user_query: str,
        query_embedding: List[float],
        bot_response: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_provider: str,
        llm_model: str,
        response_time_ms: int,
        pathway_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: str = "public"
    ) -> Optional[str]:
        """
        Log a complete query interaction to Supabase.
        
        Returns:
            query_id if successful, None if failed
        """
        if not self.client:
            print("⚠ Supabase logging skipped (not configured)")
            return None
        
        # Extract metadata from chunks
        chunk_ids = [c.get('chunk_id') for c in retrieved_chunks if 'chunk_id' in c]
        similarity_scores = [c.get('similarity_score') for c in retrieved_chunks if 'similarity_score' in c]
        
        # Check for citations in response
        has_citations = any(marker in bot_response for marker in ['[1]', '[Source:', 'Citation:'])
        
        log_entry = {
            "session_id": session_id,
            "pathway_id": pathway_id,
            "user_id": user_id or f"anon_{uuid.uuid4().hex[:8]}",
            "user_role": user_role,
            "user_query": user_query,
            "query_embedding": query_embedding,
            "bot_response": bot_response,
            "num_chunks_retrieved": len(retrieved_chunks),
            "retrieved_chunk_ids": chunk_ids,
            "avg_similarity_score": sum(similarity_scores) / len(similarity_scores) if similarity_scores else None,
            "top_similarity_scores": similarity_scores[:5] if similarity_scores else [],
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "response_time_ms": response_time_ms,
            "has_citations": has_citations,
            "num_citations": bot_response.count('[Source:') if '[Source:' in bot_response else 0
        }
        
        try:
            result = self.client.table('pathway_queries').insert(log_entry).execute()
            query_id = result.data[0]['query_id'] if result.data else None
            print(f"✓ Logged to Supabase: query_id={query_id}")
            return query_id
        except Exception as e:
            print(f"⚠ Failed to log to Supabase: {e}")
            # Don't crash the app if logging fails
            return None

# Singleton instance
query_logger = QueryLogger()