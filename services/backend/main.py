"""
MVP 1: Backend API Service
Handles chat queries, role-based routing, and RAG pipeline integration.
"""
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

# Import existing RAG components
import sys
sys.path.append('/app')
from query import get_embeddings, rag_ollama, rag_openai

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
    model: Optional[str] = "ollama"
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    timestamp: datetime
    role: str

class ChatHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: datetime

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend-api"}

# Public chat endpoint
@app.post("/chat/public", response_model=ChatResponse)
async def chat_public(request: ChatRequest):
    """
    Public chat endpoint - no authentication required.
    No memory/context stored.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get embeddings and perform RAG
        query_emb = get_embeddings([request.query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        
        # Retrieve top-k chunks
        cur.execute('''
            SELECT chunk_index, chunk_text, chunk_length, source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, request.top_k))
        
        results = cur.fetchall()
        citations = [dict(r) for r in results]
        
        # Generate response using LLM
        if request.model == "ollama":
            # Use ollama (local)
            context = "\n\n".join([f"{r['source_file']}: {r['chunk_text']}" for r in citations])
            # For now, return a simple response - integrate with rag_ollama later
            response_text = f"Based on the clinical pathways, here's information related to your query: {request.query}"
        else:
            response_text = f"Response for: {request.query}"
        
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
            timestamp=datetime.now(),
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
            SELECT chunk_index, chunk_text, chunk_length, source_file
            FROM items
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        ''', (query_emb_list, request.top_k))
        
        results = cur.fetchall()
        citations = [dict(r) for r in results]
        
        # Generate response with context
        context = "\n\n".join([f"{r['source_file']}: {r['chunk_text']}" for r in citations])
        response_text = f"Based on your previous questions and the clinical pathways: {request.query}"
        
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

