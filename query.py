from typing import List, Dict
from sentence_transformers import SentenceTransformer
import psycopg2
# import openai
from ollama import Client

# ----------------------
# Models and DB
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
ollama = Client(host="http://localhost:11434")  # default port for local server

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    return model.encode(chunk_texts)

# ----------------------
# Retrieval + RAG with OpenAI
# ----------------------
def rag_openai(cur, query: str, top_k: int = 5, model_name: str = "gpt-4"):
    """
    Retrieve top-k chunks and use OpenAI GPT to answer the query.
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
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content
    print("\n=== OpenAI Answer ===\n")
    print(answer)

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
    print(response.message.content)

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")
    cur = conn.cursor()

    query = input("Enter your query: ")
    # rag_openai(cur, query, top_k=5)
    rag_ollama(cur, query, top_k=5)
    
    conn.close()
