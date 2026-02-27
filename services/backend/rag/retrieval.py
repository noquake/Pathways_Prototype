# rag/retrieval.py
from typing import List, Dict, Optional
from .models import EMBEDDING_MODELS, DEFAULT_MODEL
from .embeddings import get_embeddings

def retrieve_chunks(supabase, query: str, top_k: int = 5, pathway_id: Optional[str] = None, model_key: str = DEFAULT_MODEL):
    """
    Retrieve chunks using specified embedding model.
    
    Args:
        supabase: Supabase client
        query: Search query
        top_k: Number of results
        pathway_id: Optional pathway filter
        model_key: Which embedding model to use
    
    Returns:
        List of matching chunks with metadata
    """
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = EMBEDDING_MODELS[model_key]
    rpc_function = config["rpc_function"]
    
    # Generate query embedding with same model
    query_emb = get_embeddings([query], model_key=model_key)[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
    
    # Call appropriate RPC function
    payload = {
        "query_embedding": query_emb_list,
        "match_count": top_k,
    }
    
    if pathway_id:
        payload["filter_pathway_id"] = pathway_id
    
    try:
        response = supabase.rpc(rpc_function, payload).execute()
        return response.data or []
    except Exception as e:
        raise RuntimeError(f"Retrieval failed for {model_key}: {e}")