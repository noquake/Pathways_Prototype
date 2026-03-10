from typing import List, Optional
from .models import EMBEDDING_MODELS, DEFAULT_MODEL
from .embeddings import get_embeddings

def retrieve_chunks(
    supabase,
    query: str,
    top_k: int = 5,
    pathway_id: Optional[str] = None,
    pathway_ids: Optional[List[str]] = None,
    model_key: str = DEFAULT_MODEL,
    rpc_function: Optional[str] = None,
    filter_arg: str = "filter_pathway_id",
):
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}")

    config = EMBEDDING_MODELS[model_key]
    rpc_name = rpc_function or config["rpc_function"]

    # is_query=True ensures MedCPT uses its Query Encoder here
    query_emb = get_embeddings([query], model_key=model_key, is_query=True)[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    payload = {
        "query_embedding": query_emb_list,
        "match_count": top_k,
    }

    try:
        if pathway_ids:
            merged_results = []
            for candidate_pathway_id in pathway_ids:
                candidate_payload = dict(payload)
                candidate_payload[filter_arg] = candidate_pathway_id
                response = supabase.rpc(rpc_name, candidate_payload).execute()
                merged_results.extend(response.data or [])

            merged_results.sort(
                key=lambda item: float(item.get("similarity", item.get("distance", 0.0))),
                reverse=True,
            )
            return merged_results[:top_k]

        if pathway_id:
            payload[filter_arg] = pathway_id

        response = supabase.rpc(rpc_name, payload).execute()
        return response.data or []
    except Exception as e:
        raise RuntimeError(f"Retrieval failed for {model_key}: {e}")
