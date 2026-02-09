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
from logger import query_logger
import time
import uuid

# Import existing RAG components
import sys
sys.path.append('/app')
from rag.query import get_embeddings, rag_ollama, rag_api_llm

app = FastAPI(title="Pathways Clinical Chat API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_db_connection():
    """Get PostgreSQL database connection."""
    db_url = os.getenv("DATABASE_URL", "postgresql://admin:password@db:5432/pathways")
    # Simple parsing for MVP 1
    if db_url.startswith("postgresql://"):
        parts = db_url.replace("postgresql://", "").split("@")
        if len(parts) == 2:
            user_pass = parts[0].split(":")
            host_port_db = parts[1].split("/")
            return psycopg2.connect(
                user=user_pass[0],
                password=user_pass[1],
                host=host_port_db[0].split(":")[0],
                port=int(host_port_db[0].split(":")[1]) if ":" in host_port_db[0] else 5432,
                dbname=host_port_db[1].split("?")[0]
            )
    # Fallback
    return psycopg2.connect(
        dbname="pathways",
        user="admin",
        password="password",
        host="db",
        port=5432
    )

# JWT validation
async def verify_token(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Verify JWT token and extract user info."""
    if not authorization:
        return None
    
    try:
        token = authorization.replace("Bearer ", "")
        # In production, verify with Keycloak public key
        # For MVP 1, we'll use a simple JWT decode
        payload = jwt.decode(token, os.getenv("JWT_SECRET", "change-me"), algorithms=["HS256"])
        return payload
    except JWTError:
        return None

def get_user_role(user_info: Optional[Dict[str, Any]] = Depends(verify_token)) -> str:
    """Extract user role from JWT or return 'public'."""
    if user_info and "role" in user_info:
        return user_info["role"]
    return "public"

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = "gemini"  # Changed default from "ollama" to "gemini"
    model_name: Optional[str] = "gemini-2.5-flash"  # Specific Gemini model to use (2026 API)
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    timestamp: str
    role: str

class ChatHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend-api"}

# Public chat endpoint
@app.post("/chat/public", response_model=ChatResponse)
async def chat_public(request: ChatRequest):

    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        print("="*60)
        print("=== NEW QUERY RECEIVED ===")
        print(f"Session ID: {session_id}")
        print(f"Query: {request.query}")
        print(f"Model: {request.model}")
        print(f"Top K: {request.top_k}")
        print("="*60)

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n[1/6] Generating embeddings...")
        query_emb = get_embeddings([request.query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        print(f"✓ Embedding generated: dimension={len(query_emb_list)}\n")
        
        # Retrieve top-k chunks
        print(f"\n[2/6] Retrieving top {request.top_k} chunks...")
        cur.execute('''
            SELECT chunk_id, chunk_text, chunk_length, doc_name as source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, request.top_k))
        
        results = cur.fetchall()
        print(f"DEBUG: Retrieved {len(results)} results\n")

        citations = []
        for r in results:
            citation = {
                "chunk_id": int(r['chunk_id']),
                "chunk_text": str(r['chunk_text']),
                "chunk_length": int(r['chunk_length']) if r['chunk_length'] is not None else 0,
                "source_file": str(r['source_file']) if r['source_file'] else "",
                "similarity_score": float(r['similarity_score'])
            }
            citations.append(citation)
        
        if citations:
            print(f"\nTop chunk:")
            print(f"  - Source: {citations[0]['source_file']}")
            print(f"  - Similarity: {citations[0]['similarity_score']:.4f}")
        
        print(f"\n[3/6] Sending to {request.model}...")
        # Generate response using LLM
        if request.model == "gemini":
            # Use Gemini API (default)
            response_text = rag_api_llm(cur, request.query, top_k=request.top_k, 
                                       model_name=request.model_name, api_provider="gemini")
        else:
            # Default to Gemini if unknown model specified
            response_text = rag_api_llm(cur, request.query, top_k=request.top_k, 
                                       model_name="gemini-2.5-flash", api_provider="gemini")
        
        print(f"\n[4/6] Response received:")
        
        # Log public query (anonymized)
        cur.execute('''
            INSERT INTO chat_logs_public (query, response, timestamp)
            VALUES (%s, %s, %s)
        ''', (request.query[:100], response_text[:500], datetime.now()))
        conn.commit()
        
        cur.close()
        conn.close()
        
        return ChatResponse(
            response=response_text,
            citations=citations,
            timestamp=datetime.now().isoformat(),
            role="public"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Practitioner chat endpoint
@app.post("/chat/practitioner", response_model=ChatResponse)
async def chat_practitioner(
    request: ChatRequest,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """
    Practitioner chat endpoint - requires authentication.
    Includes memory/context from previous interactions.
    """
    if not user_info or user_info.get("role") != "practitioner":
        raise HTTPException(status_code=403, detail="Access denied. Practitioner role required.")
    
    user_id = user_info.get("user_id")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Retrieve recent conversation history
        cur.execute('''
            SELECT query, response, timestamp
            FROM chat_logs_practitioner
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 5
        ''', (user_id,))
        history = cur.fetchall()
        
        # Build context from history
        context_history = "\n\nPrevious conversation:\n"
        for h in reversed(history):
            context_history += f"Q: {h['query']}\nA: {h['response'][:200]}...\n\n"
        
        # Perform RAG
        query_emb = get_embeddings([request.query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        
        cur.execute('''
            SELECT chunk_id, chunk_text, chunk_length, doc_name as source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, request.top_k))
        
        results = cur.fetchall()
        citations = [dict(r) for r in results]
        
        # Generate response with context using specified model
        if request.model == "ollama":
            response_text = rag_ollama(cur, request.query, top_k=request.top_k)
        elif request.model == "gemini":
            response_text = rag_api_llm(cur, request.query, top_k=request.top_k, 
                                       model_name=request.model_name, api_provider="gemini")
        else:
            # Default to Gemini
            response_text = rag_api_llm(cur, request.query, top_k=request.top_k, 
                                       model_name="gemini-2.5-flash", api_provider="gemini")
        
        # Store in practitioner memory
        cur.execute('''
            INSERT INTO chat_logs_practitioner (user_id, query, response, timestamp)
            VALUES (%s, %s, %s, %s)
        ''', (user_id, request.query, response_text, datetime.now()))
        conn.commit()
        
        cur.close()
        conn.close()
        
        return ChatResponse(
            response=response_text,
            citations=citations,
            timestamp=datetime.now(),
            role="practitioner"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get practitioner history
@app.get("/history/{user_id}", response_model=List[ChatHistoryItem])
async def get_history(
    user_id: str,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Get chat history for a practitioner."""
    if not user_info or (user_info.get("user_id") != user_id and user_info.get("role") not in ["admin", "hr"]):
        raise HTTPException(status_code=403, detail="Access denied.")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT query, response, timestamp
            FROM chat_logs_practitioner
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (user_id,))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return [ChatHistoryItem(**dict(r)) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

