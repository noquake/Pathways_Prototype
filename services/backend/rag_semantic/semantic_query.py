from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import openai
import os
from google import genai
import ollama
from ollama import Client

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
model: Optional[SentenceTransformer] = None
ollama_client = Client(host="http://localhost:11434")  # default port for local Ollama server

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    global model
    if model is None:
        # Prefer offline/local cache to keep service startup independent of network.
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    return model.encode(chunk_texts)

def retrieve_chunks(supabase, query_emb_list, top_k: int = 5, pathway_id: str = None):
    """Retrieve top-k chunks from Supabase RPC function."""
    configured_rpc = os.getenv("SUPABASE_MATCH_RPC", "match_semantic_pathway_chunks")
    rpc_candidates = [configured_rpc, "match_semantic_pathway_chunks", "match_documents", "match_chunks"]
    # Keep deterministic order while removing duplicates.
    rpc_names = list(dict.fromkeys(rpc_candidates))

    rpc_payload = {
        "query_embedding": query_emb_list,
        "match_count": top_k,
    }
    # If SQL function supports pathway filtering, pass it through.
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
        f"Supabase retrieval RPC failed. Tried: {', '.join(rpc_names)}. Last error: {last_error}"
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

# ----------------------
# Retrieval + RAG with API-based LLMs
# ----------------------
def rag_api_llm(supabase, query: str, top_k: int = 5, model_name: str = "gpt-4", api_provider: str = "gemini", pathway_id: str = None, retrieved_results=None):
    """
    Retrieve top-k chunks and use an API-based LLM (OpenAI, Gemini, etc.) to answer the query.
    
    Args:
        supabase: Supabase client
        query: User query string
        top_k: Number of top chunks to retrieve
        model_name: Name of the model to use
        api_provider: API provider to use ("openai" or "gemini")
    """
    results = retrieved_results
    if results is None:
        query_emb = get_embeddings([query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        results = retrieve_chunks(supabase, query_emb_list, top_k=top_k, pathway_id=pathway_id)

    if not results:
        print("No relevant chunks found.")
        return "No relevant chunks found in pathway documents."

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
    
    if api_provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content
        print("\n=== OpenAI Answer ===\n")
        print(answer)
        return answer
    elif api_provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Create Gemini client with API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Map model names to correct format for new API (2026 models)
        model_mapping = {
            "gemini-1.5-flash": "gemini-2.5-flash",
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-pro": "gemini-2.5-flash",  # Map old name to new model
            "gemini-pro-vision": "gemini-2.5-pro",
            "gemini-flash-latest": "gemini-flash-latest",
            "gemini-pro-latest": "gemini-pro-latest",
        }
        
        # Get the correct model name
        if model_name.startswith("gpt"):
            model_name = "gemini-2.5-flash"  # Default for OpenAI model names
        
        actual_model = model_mapping.get(model_name, "gemini-2.5-flash")
        
        # Generate response using new API
        response = client.models.generate_content(
            model=actual_model,
            contents=prompt
        )
        
        # Extract answer
        answer = response.text
        print("\n=== Gemini Answer ===\n")
        print(answer)
        return answer
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'gemini'.")

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
    # Use Gemini by default (with gemini-2.5-flash for faster responses)
    rag_api_llm(supabase, query, top_k=5, model_name="gemini-2.5-flash", api_provider="gemini")