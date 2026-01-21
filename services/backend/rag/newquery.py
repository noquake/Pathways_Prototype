from typing import List, Dict
from sentence_transformers import SentenceTransformer
import psycopg2
import os
import google.generativeai as gemini
import ollama 

# Optional: Import OpenAI if you plan to use it
try:
    import openai
except ImportError:
    openai = None

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
# Load the embedding model once when the script starts
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)

# ----------------------
# Retrieval + RAG with API-based LLMs (Gemini / OpenAI)
# ----------------------
def rag_api_llm(cur, query: str, top_k: int = 5, model_name: str = "gemini-1.5-flash", api_provider: str = "gemini"):
    """
    Retrieve top-k chunks and use an API-based LLM to answer the query.
    """
    # 1. Compute query embedding
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    # 2. Retrieve top-k chunks
    cur.execute('''
        SELECT chunk_text, source_file
        FROM items
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    ''', (query_emb_list, top_k))
    
    results = cur.fetchall()
    if not results:
        return "No relevant clinical pathways found to answer your query."

    # 3. Build Context
    context = "\n\n".join([f"Source ({r[1]}): {r[0]}" for r in results])
    
    prompt = f"""
    You are a clinical assistant. Use the following context to answer the question strictly.
    If the answer is not in the context, say you do not know.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    
    # 4. Generate Response based on Provider
    if api_provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini
        gemini.configure(api_key=GEMINI_API_KEY)
        
        # Smart Model Fallback: If user asks for GPT but uses Gemini provider, switch to Gemini
        valid_gemini_models = ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
        if model_name not in valid_gemini_models:
            model_name = "gemini-1.5-flash" # Default to the faster/cheaper one
        
        # Create model and generate
        gemini_model = gemini.GenerativeModel(model_name)
        response = gemini_model.generate_content(prompt)
        
        return response.text

    elif api_provider.lower() == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not openai:
            raise ImportError("OpenAI library not installed.")
            
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model=model_name if model_name.startswith("gpt") else "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")


# def rag_ollama(cur, query: str, top_k: int = 5, model_name: str = "llama3"):
#     """
#     Retrieve top-k chunks and use local Ollama to answer the query.
#     """
#     # 1. Compute query embedding
#     query_emb = get_embeddings([query])[0]
#     query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

#     # 2. Retrieve top-k chunks
#     cur.execute('''
#         SELECT chunk_text, source_file
#         FROM items
#         ORDER BY embedding <-> %s::vector
#         LIMIT %s
#     ''', (query_emb_list, top_k))
    
#     results = cur.fetchall()
#     if not results:
#         return "No relevant clinical pathways found."

#     context = "\n\n".join([f"Source ({r[1]}): {r[0]}" for r in results])
    
#     prompt = f"""
#     Use the following clinical context to answer the question.

#     Context:
#     {context}

#     Question:
#     {query}

#     Answer:
#     """

#     # 3. Call Ollama (Using the correct function syntax)
#     # Note: Ensure you have pulled the model first: `ollama pull llama3`
#     response = ollama.chat(
#         model=model_name, 
#         messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response['message']['content']
