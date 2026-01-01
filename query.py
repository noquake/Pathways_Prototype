from typing import List, Dict
from sentence_transformers import SentenceTransformer
import psycopg2
import openai
import os
import google.generativeai as gemini
from ollama import Ollama
from ollama import Client

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------
# Models and DB
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
ollama = Ollama()  # defaults to localhost
ollama = Client(host="http://localhost:11434")  # default port for local server

response = ollama.chat(model="llama2", messages=[{"role": "user", "content": "Hello, Ollama!"}])
print(response['content'])

ollama = Ollama()  # Only if using local LLaMA

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)

# ----------------------
# Retrieval + RAG with API-based LLMs
# ----------------------
def rag_api_llm(cur, query: str, top_k: int = 5, model_name: str = "gpt-4", api_provider: str = "gemini"):
    """
    Retrieve top-k chunks and use an API-based LLM (OpenAI, Gemini, etc.) to answer the query.
    
    Args:
        cur: Database cursor
        query: User query string
        top_k: Number of top chunks to retrieve
        model_name: Name of the model to use
        api_provider: API provider to use ("openai" or "gemini")
    """
    # Compute query embedding
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    # Retrieve top-k chunks
    cur.execute('''
        SELECT chunk_text, source_file
        FROM items
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    ''', (query_emb_list, top_k))
    
    results = cur.fetchall()
    if not results:
        print("No relevant chunks found.")
        return

    context = "\n\n".join([f"{r[1]}: {r[0]}" for r in results])
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

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
    elif api_provider.lower() == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        # Configure Gemini API
        gemini.configure(api_key=GEMINI_API_KEY)
        
        # Get the model (default to gemini-pro if model_name not specified or is OpenAI model)
        if model_name.startswith("gpt") or model_name not in ["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro", "gemini-1.5-flash"]:
            # Default to gemini-pro if an OpenAI model name is provided
            model_name = "gemini-pro"
        
        # Create the model instance
        gemini_model = gemini.GenerativeModel(model_name)
        
        # Generate response
        response = gemini_model.generate_content(prompt)
        
        # Extract answer
        answer = response.text
        print("\n=== Gemini Answer ===\n")
        print(answer)
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'gemini'.")

# ----------------------
# Retrieval + RAG with local LLaMA
# ----------------------
def rag_ollama(cur, query: str, top_k: int = 5, model_name: str = "llama2"):
    """
    Retrieve top-k chunks and use local LLaMA (Ollama) to answer the query.
    """
    query_emb = get_embeddings([query])[0]
    query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb

    # Retrieve top-k chunks
    cur.execute('''
        SELECT chunk_text, source_file
        FROM items
        ORDER BY embedding <-> %s::vector
        LIMIT %s
    ''', (query_emb_list, top_k))
    
    results = cur.fetchall()
    if not results:
        print("No relevant chunks found.")
        return

    context = "\n\n".join([f"{r[1]}: {r[0]}" for r in results])
    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    print("\n=== Local LLaMA Answer ===\n")
    print(response['content'])

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")
    cur = conn.cursor()

    query = input("Enter your query: ")
    # rag_api_llm(cur, query, top_k=5, api_provider="openai")
    rag_ollama(cur, query, top_k=5)
