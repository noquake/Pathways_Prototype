from typing import List, Dict
from sentence_transformers import SentenceTransformer
from supabase import Client
import openai
import os
from google import genai

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)

# ----------------------
# Retrieval + RAG with API-based LLMs
# ----------------------
def rag_api_llm(supabase: Client, query: str, top_k: int = 5, model_name: str = "gpt-4", api_provider: str = "gemini"):
    """
    Retrieve top-k chunks and use an API-based LLM (OpenAI, Gemini, etc.) to answer the query.
    
    Args:
        supabase: Supabase client
    """
    # Compute query embedding
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    # Retrieve top-k chunks using Supabase RPC function
    response = supabase.rpc(
        'match_pathway_chunks',
        {
            'query_embedding': query_emb_list,
            'match_count': top_k
        }
    ).execute()
    
    results = response.data
    
    if not results:
        print("No relevant chunks found.")
        return "No relevant information found in the knowledge base."

    # Build context from results
    context = "\n\n".join([
        f"[{i+1}] {r['pathway_id']}: {r['chunk_text']}" 
        for i, r in enumerate(results)
    ])
    
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

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))
    query = input("Enter your query: ")
    # Use Gemini by default (with gemini-2.5-flash for faster responses)
    rag_api_llm(supabase, query, top_k=5, model_name="gemini-2.5-flash", api_provider="gemini")
