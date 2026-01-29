from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import os
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from jose import JWTError, jwt
import httpx
import sys

from google import genai

# Add the app directory to sys.path so we can import from rag folder
sys.path.append('/app')

# --- IMPORT YOUR NEW LOGIC HERE ---
# We import rag_api_llm from the file you uploaded (testquery.py)
# Make sure testquery.py is inside the /rag folder
from rag.testquery import rag_api_llm, get_embeddings

client = genai.Client()

app = FastAPI(title="Pathways Clinical Chat API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    # Allow your frontend running on localhost:3000 (React)
    allow_origins=["http://localhost:3000", "http://frontend:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_connection():
    """Get PostgreSQL database connection."""
    try:
        # Use 'pathways_db' (service name) as the default host for Docker
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "pathways"),
            user=os.getenv("POSTGRES_USER", "admin"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            host=os.getenv("DB_HOST", "pathways_db"),
            port=5432
        )
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = "gemini-3-flash-preview" 
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    timestamp: datetime
    role: str

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend-api"}

# --- UPDATED PUBLIC CHAT ENDPOINT ---
@app.post("/chat/public", response_model=ChatResponse)
async def chat_public(request: ChatRequest):
    """
    Public chat endpoint - Uses Gemini RAG
    """
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # --- CALL YOUR RAG LOGIC ---
        # We pass "gemini" as the provider to force the API usage
        result = rag_api_llm(
            cur=cur, 
            query=request.query, 
            top_k=request.top_k, 
            model_name="gemini-3-flash-preview",
            api_provider="gemini" 
        )
        
        # Log public query (anonymized)
        # Note: We use a try-except here so logging failures don't break the user experience
        try:
            cur.execute('''
                INSERT INTO chat_logs_public (query, response, timestamp)
                VALUES (%s, %s, %s)
            ''', (request.query[:100], result["answer"][:500], datetime.now()))
            conn.commit()
        except Exception as log_error:
            print(f"Warning: Could not log chat: {log_error}")
            conn.rollback() # Rollback the failed log insert but keep the connection for the close
        
        cur.close()
        conn.close()
        
        return ChatResponse(
            response=result["answer"],
            citations=result["citations"],
            timestamp=datetime.now(),
            role="public"
        )
    except Exception as e:
        if conn: conn.close()
        print(f"Error in chat_public: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" allows external access 
    uvicorn.run(app, host="0.0.0.0", port=8000)