from dotenv import load_dotenv
load_dotenv()

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
from pathways_catalog import list_pathways, get_pathway_by_id


# Import existing RAG components
import sys
sys.path.append('/app')
from rag.query import get_embeddings, rag_api_llm

app = FastAPI(title="Pathways Clinical Chat API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://frontend:3000",
    ],
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
    model_name: Optional[str] = "gemini-2.5-flash"  # Specific Gemini model to use (2026 API)
    top_k: Optional[int] = 5
    pathway_id: Optional[str] = None


class PathwayOption(BaseModel):
    id: str
    label: str
    doc_name: str
    preview_image_path: str

# class ChatResponse(BaseModel):
#     response: str
#     citations: List[Dict[str, Any]]
#     timestamp: str
#     role: str

class Citation(BaseModel):
    chunk_id: str
    chunk_text: str
    chunk_length: int
    source_file: str

class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    timestamp: str
    role: str

class ChatHistoryItem(BaseModel):
    query: str
    response: str
    timestamp: str


def resolve_pathway_doc_name(pathway_id: Optional[str]) -> Optional[str]:
    if not pathway_id:
        return None

    pathway = get_pathway_by_id(pathway_id)
    if not pathway:
        raise HTTPException(status_code=400, detail=f"Unknown pathway_id: {pathway_id}")
    return pathway["doc_name"]

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend-api"}


@app.get("/pathways", response_model=List[PathwayOption])
async def get_pathways():
    """Return curated pathway options for the frontend dropdown."""
    return [PathwayOption(**pathway) for pathway in list_pathways()]

# Public chat endpoint
@app.post("/chat/public", response_model=ChatResponse)
async def chat_public(request: ChatRequest):

    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        selected_pathway_doc_name = resolve_pathway_doc_name(request.pathway_id)

        print("="*60)
        print("=== NEW QUERY RECEIVED ===")
        print(f"Session ID: {session_id}")
        print(f"Query: {request.query}")
        print("Provider: gemini")
        print(f"Model: {request.model_name}")
        print(f"Top K: {request.top_k}")
        if selected_pathway_doc_name:
            print(f"Pathway filter requested: {request.pathway_id} -> {selected_pathway_doc_name}")
        print("="*60)

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        print("\n[1/6] Generating embeddings...")
        query_emb = get_embeddings([request.query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        print(f"✓ Embedding generated: dimension={len(query_emb_list)}\n")
        
        # Retrieve top-k chunks
        print(f"\n[2/6] Retrieving top {request.top_k} chunks...")
        if selected_pathway_doc_name:
            cur.execute('''
            SELECT 
            chunk_id, 
            chunk_text, 
            chunk_length, 
            doc_name as source_file,
            (embedding <-> %s::vector) as distance
            FROM items
            WHERE doc_name = %s
            ORDER BY distance
            LIMIT %s
            ''', (query_emb_list, selected_pathway_doc_name, request.top_k))
        else:
            cur.execute('''
            SELECT 
            chunk_id, 
            chunk_text, 
            chunk_length, 
            doc_name as source_file,
            (embedding <-> %s::vector) as distance
            FROM items
            ORDER BY distance
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
                "similarity_score": float(r['distance'])  # ← Changed this line
            }
            citations.append(citation)

        if citations:
            print(f"\nTop chunk:")
            print(f"  - Source: {citations[0]['source_file']}")
            print(f"  - Similarity: {citations[0]['similarity_score']:.4f}")
        
        print("\n[3/6] Sending to gemini...")
        response_text = rag_api_llm(
            cur,
            request.query,
            top_k=request.top_k,
            model_name=request.model_name,
            doc_name_filter=selected_pathway_doc_name,
        )

        if not isinstance(response_text, str) or not response_text.strip():
            response_text = "I couldn't generate a response right now. Please try again."
        
        print(f"\n[4/6] Response received:")
        print(f"✓ Length: {len(response_text)} chars")
        print(f"✓ Preview: {response_text[:150]}...")

        response_time_ms = int((time.time() - start_time) * 1000)

        print(f"\n[5/6] Logging to Supabase...")
        query_logger.log_query(
            session_id=session_id,
            user_query=request.query,
            query_embedding=query_emb_list,
            bot_response=response_text,
            retrieved_chunks=citations,
            llm_provider="gemini",
            llm_model=request.model_name,
            response_time_ms=response_time_ms,
            pathway_id=request.pathway_id,
            user_role="public"
        )
        
        cur.close()
        conn.close()

        print(f"\n[6/6] Creating response...")
        
        # different citations for end users, omitting the similarity score
        client_citations = [
            {
                "chunk_id": str(c['chunk_id']),
                "chunk_text": c['chunk_text'],
                "chunk_length": c['chunk_length'],
                "source_file": c['source_file']
            }
            for c in citations
        ]
        
        response_obj = ChatResponse(
            response=response_text,
            citations=client_citations,
            timestamp=datetime.now().isoformat(),
            role="public"
        )

        print(f"✓ Response time: {response_time_ms}ms")
        print("="*60)
        print("=== QUERY COMPLETE ===")
        print("="*60)

        return response_obj
    
    except Exception as e:
        print("\n" + "!"*60)
        print("!!! ERROR OCCURRED !!!")
        print(f"Query: {request.query}")
        print(f"Error: {e}")
        print("!"*60)
        import traceback
        traceback.print_exc()
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
    
    selected_pathway_doc_name = resolve_pathway_doc_name(request.pathway_id)
    user_id = user_info.get("user_id")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if selected_pathway_doc_name:
            print(f"Pathway filter requested: {request.pathway_id} -> {selected_pathway_doc_name}")
        
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
        
        if selected_pathway_doc_name:
            cur.execute('''
                SELECT chunk_id, chunk_text, chunk_length, doc_name as source_file
                FROM items
                WHERE doc_name = %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            ''', (selected_pathway_doc_name, query_emb_list, request.top_k))
        else:
            cur.execute('''
                SELECT chunk_id, chunk_text, chunk_length, doc_name as source_file
                FROM items
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            ''', (query_emb_list, request.top_k))
        
        results = cur.fetchall()
        citations = [dict(r) for r in results]
        
        # Generate response using Gemini API only
        response_text = rag_api_llm(
            cur,
            request.query,
            top_k=request.top_k,
            model_name=request.model_name,
            doc_name_filter=selected_pathway_doc_name,
        )
        if not isinstance(response_text, str) or not response_text.strip():
            response_text = "I couldn't generate a response right now. Please try again."
        
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
