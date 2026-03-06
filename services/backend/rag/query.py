from typing import List, Optional
from sentence_transformers import SentenceTransformer
import os
from google import genai
from ollama import Client

from rag.embeddings import get_embeddings

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
ollama_client = Client(host="http://localhost:11434")  # default port for local Ollama server

def retrieve_chunks(supabase, query_emb_list, top_k: int = 5, pathway_id: str = None):
    """Retrieve top-k chunks from Supabase RPC function."""
    configured_rpc = os.getenv("SUPABASE_MATCH_RPC", "match_pathway_chunks")
    rpc_candidates = [configured_rpc, "match_pathway_chunks", "match_documents", "match_chunks"]
    rpc_names = list(dict.fromkeys(rpc_candidates))

    rpc_payload = {
        "query_embedding": query_emb_list,
        "match_count": top_k,
    }
    if pathway_id:
        rpc_payload["filter_pathway_id"] = pathway_id

    last_error = None
    for rpc_name in rpc_names:
        try:
            response = supabase.rpc(rpc_name, rpc_payload).execute()
            return response.data or []
        except Exception as err:
            last_error = err
            # Backward-compatible fallback when RPC doesn't accept filter argument.
            if pathway_id and "filter_pathway_id" in rpc_payload:
                try:
                    payload_without_filter = dict(rpc_payload)
                    payload_without_filter.pop("filter_pathway_id", None)
                    response = supabase.rpc(rpc_name, payload_without_filter).execute()
                    data = response.data or []
                    return [row for row in data if row.get("pathway_id") == pathway_id]
                except Exception as inner_err:
                    last_error = inner_err
                    continue
            continue

    raise RuntimeError(
        f"Supabase retrieval RPC failed for DEFAULT chunks. Tried: {', '.join(rpc_names)}. Last error: {last_error}"
    )


def build_context(results):
    """Build prompt context string from retrieval results."""
    if results and isinstance(results[0], dict):
        context_lines = []
        for i, r in enumerate(results):
            source = r.get("source_file") or r.get("pathway_id") or "unknown"
            text = r.get("chunk_text", "")
            context_lines.append(f"[{i+1}] {source}: {text}")
        return "\n\n".join(context_lines)

    context_lines = []
    for i, r in enumerate(results):
        # retrieve_chunks tuple shape: (chunk_id, chunk_text, chunk_length, source_file, distance)
        if isinstance(r, (tuple, list)) and len(r) >= 4:
            source = r[3]
            text = r[1]
        # backward compatibility for older tuple shape: (chunk_text, source_file)
        elif isinstance(r, (tuple, list)) and len(r) >= 2:
            source = r[1]
            text = r[0]
        else:
            source = "unknown"
            text = str(r)
        context_lines.append(f"[{i+1}] {source}: {text}")
    return "\n\n".join(context_lines)


def extract_gemini_text(response) -> Optional[str]:
    """Best-effort extraction for Gemini responses that may not populate `response.text`."""
    direct_text = getattr(response, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    candidates = getattr(response, "candidates", None) or []
    candidate_parts: List[str] = []

    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        for part in parts or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                candidate_parts.append(part_text.strip())

    if candidate_parts:
        return "\n\n".join(candidate_parts)

    return None


# ----------------------
# Retrieval + RAG with Gemini API
# ----------------------
def rag_api_llm(
    supabase,
    query: str,
    top_k: int = 5,
    model_name: str = "gemini-2.5-flash",
    pathway_id: Optional[str] = None,
    retrieved_results=None,
):
    """
    Retrieve top-k chunks and use Gemini API to answer the query.

    Args:
        supabase: Supabase client
        query: User query string
        top_k: Number of top chunks to retrieve
        model_name: Gemini model name to use
        pathway_id: Optional pathway filter
        retrieved_results: Optional pre-retrieved chunks to avoid duplicate retrieval
    """
    results = retrieved_results
    if results is None:
        query_emb = get_embeddings([query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        results = retrieve_chunks(supabase, query_emb_list, top_k=top_k, pathway_id=pathway_id)

    if not results:
        print("No relevant chunks found.")
        return "I couldn't find relevant pathway content for this query."

    context = build_context(results)

    prompt = f"""
You are a clinical assistant that STRICTLY follows institutional protocols.

CRITICAL INSTRUCTIONS:
- You MUST answer ONLY based on the provided context sources below
- The context sources are AUTHORITATIVE clinical protocols that you must follow exactly
- DO NOT use any external knowledge or add caveats about missing information
- DO NOT say "the context doesn't contain" - if information appears in the sources or their references, state it definitively
- When information is in the sources, present it as established medical protocol
- ALWAYS cite sources using [1], [2], etc. when referencing information

Context Sources (AUTHORITATIVE INSTITUTIONAL PROTOCOLS):
{context}

Question:
{query}

Provide a definitive answer based ONLY on the context above. Present the information from the sources as established protocol without hedging or adding external caveats.

Answer:
"""

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=GEMINI_API_KEY)

    model_mapping = {
        "gemini-1.5-flash": "gemini-2.5-flash",
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-pro": "gemini-2.5-flash",
        "gemini-pro-vision": "gemini-2.5-pro",
        "gemini-flash-latest": "gemini-flash-latest",
        "gemini-pro-latest": "gemini-pro-latest",
    }

    actual_model = model_mapping.get(model_name, "gemini-2.5-flash")

    response = client.models.generate_content(
        model=actual_model,
        contents=prompt,
    )

    answer = extract_gemini_text(response)
    if not answer:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
        print("\n=== Gemini Answer ===\n")
        print(None)
        print(f"Gemini returned empty text. block_reason={block_reason}")
        return "I couldn't generate a complete response for that request. Please rephrase and try again."

    print("\n=== Gemini Answer ===\n")
    print(answer)
    return answer


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    from supabase import create_client
    from dotenv import load_dotenv

    load_dotenv()
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY"),
    )

    query = input("Enter your query: ")
    rag_api_llm(supabase, query, top_k=5, model_name="gemini-2.5-flash")
