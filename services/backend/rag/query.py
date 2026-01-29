import os
import psycopg2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# 1. Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Fallback for local testing if env var isn't set
    print("Warning: GEMINI_API_KEY not found in environment.")

# 2. Initialize Models (Load once to save time)
# Note: In production, you might want to load the embedding model globally or via a singleton
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(text: str) -> List[float]:
    """Generate embedding for a single query string."""
    # model.encode returns a numpy array, we need a list for pgvector
    return embed_model.encode(text).tolist()

def get_db_connection():
    """Connect to the database using Docker network credentials."""
    try:
        # We use 'pathways_db' as the host because that is the service name in docker-compose
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "pathways"),
            user=os.getenv("POSTGRES_USER", "admin"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            host=os.getenv("DB_HOST", "pathways_db"), # Defaults to Docker service name
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def query_gemini(context: str, query: str):
    """Send the context and query to Gemini."""
    genai.configure(api_key=GEMINI_API_KEY)
    
    # gemini-1.5-flash is faster and cheaper for RAG; use 1.5-pro for complex reasoning
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    You are a helpful assistant for a clinical pathway documentation system.
    Use the following retrieved context to answer the user's question.
    If the answer is not in the context, say "I cannot find the answer in the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response from Gemini: {e}"

def rag_pipeline(query: str, top_k: int = 5):
    """Main function to retrieve data and generate answer."""
    
    # 1. Embed the query
    query_vector = get_embeddings(query)

    # 2. Retrieve relevant chunks from Postgres
    conn = get_db_connection()
    if not conn:
        return "Database connection failed."
    
    try:
        cur = conn.cursor()
        # The <-> operator is L2 distance (Euclidean). 
        # For cosine similarity (if normalized), use <=> 
        sql = """
            SELECT chunk_text, source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (query_vector, top_k))
        results = cur.fetchall()
        cur.close()
        conn.close()

        if not results:
            return "No relevant documents found in the database."

        # 3. Format Context
        # We combine the source file name with the text for better context
        context_str = "\n\n".join([f"Source ({r[1]}): {r[0]}" for r in results])

        # 4. Generate Answer
        print(f"\n--- Retrieved {len(results)} chunks. Generating answer... ---\n")
        answer = query_gemini(context_str, query)
        return answer

    except Exception as e:
        if conn:
            conn.close()
        return f"Error during RAG pipeline: {e}"

if __name__ == "__main__":
    # Simple CLI loop for testing inside the container
    print("RAG System Ready (Gemini Powered). Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your query: ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        response = rag_pipeline(user_query)
        print("\n=== Gemini Response ===")
        print(response)
        print("=======================")