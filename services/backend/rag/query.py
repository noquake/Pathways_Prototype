from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import psycopg2
import os
from google import genai

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)


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
# Retrieval + RAG with API-based LLMs
# ----------------------
def rag_api_llm(
    cur,
    query: str,
    top_k: int = 5,
    model_name: str = "gemini-2.5-flash",
    doc_name_filter: Optional[str] = None,
):
    """
    Retrieve top-k chunks and use Gemini API to answer the query.
    
    Args:
        cur: Database cursor
        query: User query string
        top_k: Number of top chunks to retrieve
        model_name: Gemini model name to use
    """
    # Compute query embedding
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    # Retrieve top-k chunks
    if doc_name_filter:
        cur.execute('''
            SELECT chunk_text, doc_name as source_file
            FROM items
            WHERE doc_name = %s
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (doc_name_filter, query_emb_list, top_k))
    else:
        cur.execute('''
            SELECT chunk_text, doc_name as source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, top_k))
    
    results = cur.fetchall()
    if not results:
        print("No relevant chunks found.")
        return "I couldn't find relevant pathway content for this query."

    # Handle both tuple and dict results from cursor
    if results and isinstance(results[0], dict):
        context = "\n\n".join([f"[{i+1}] {r['source_file']}: {r['chunk_text']}" for i, r in enumerate(results)])
    else:
        context = "\n\n".join([f"[{i+1}] {r[1]}: {r[0]}" for i, r in enumerate(results)])
    
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
    
    actual_model = model_mapping.get(model_name, "gemini-2.5-flash")
    
    # Generate response using Gemini API
    response = client.models.generate_content(
        model=actual_model,
        contents=prompt
    )
    
    # Extract answer with fallback logic for safety-blocked/non-text responses.
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
    conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")
    cur = conn.cursor()

    query = input("Enter your query: ")
    # Use Gemini by default (with gemini-2.5-flash for faster responses)
    rag_api_llm(cur, query, top_k=5, model_name="gemini-2.5-flash")
