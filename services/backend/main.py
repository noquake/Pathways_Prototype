from dotenv import load_dotenv
load_dotenv()


load_dotenv()
from httpx import request
from fastapi import FastAPI, HTTPException, Depends, Header, Response
import sys
sys.path.append('/app')
import os
from datetime import datetime
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from supabase import create_client
from jose import JWTError, jwt

from logger import query_logger
from pathways_catalog import list_pathways, get_pathway_by_id, get_pathway_resource

# Import existing RAG components
from rag.embeddings import get_embeddings

from rag.query import rag_api_llm as original_rag_api_llm

from rag.query import rag_api_llm as original_rag_api_llm
from rag.retrieval import retrieve_chunks as retrieve_chunks_by_model

app = FastAPI(title="Pathways Clinical Chat API", version="1.0.0")

PATHWAY_QUERIES_TABLE = os.getenv("SUPABASE_TABLE_PATHWAY_QUERIES", "pathway_queries")

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


def get_supabase_client():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
    )
    if not supabase_url or not supabase_key:
        raise ValueError(
            "Supabase credentials are missing. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY / SUPABASE_PUBLISHABLE_KEY)."
        )
    return create_client(supabase_url, supabase_key)


# JWT validation
async def verify_token(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Verify JWT token and extract user info."""
    if not authorization:
        return None

    try:
        token = authorization.replace("Bearer ", "")
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
    model: Optional[str] = "gemini"
    model_name: Optional[str] = "gemini-2.5-flash"
    top_k: Optional[int] = 5
    pathway_id: Optional[str] = None
    embedding_model: Optional[str] = "minilm"  # ← add this


class PractitionerChatRequest(ChatRequest):
    pathway_id: str


class PathwayResourceOption(BaseModel):
    id: str
    label: str
    doc_name: str
    pdf_url: str


class PathwayOption(BaseModel):
    id: str
    label: str
    default_resource_id: str
    doc_name: str
    pdf_url: str
    resources: List[PathwayResourceOption]


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


@app.get("/pathways/{pathway_id}/pdf")
async def get_pathway_pdf(
    pathway_id: str,
    resource_id: Optional[str] = None,
    range_header: Optional[str] = Header(None, alias="Range"),
    if_range: Optional[str] = Header(None, alias="If-Range"),
):
    pathway = get_pathway_by_id(pathway_id)
    if not pathway:
        raise HTTPException(status_code=404, detail=f"Unknown pathway_id: {pathway_id}")

    resource = get_pathway_resource(pathway_id, resource_id)
    if not resource:
        raise HTTPException(status_code=404, detail=f"Unknown resource_id for pathway_id: {pathway_id}")

    upstream_headers = {}
    if range_header:
        upstream_headers["Range"] = range_header
    if if_range:
        upstream_headers["If-Range"] = if_range

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(20.0, connect=10.0),
        ) as client:
            upstream_response = await client.get(resource["pdf_url"], headers=upstream_headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Timed out fetching pathway PDF.")
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Could not fetch pathway PDF.")

    if upstream_response.status_code not in (200, 206):
        raise HTTPException(status_code=502, detail="Upstream pathway PDF request failed.")

    content_type = upstream_response.headers.get("content-type", "").lower()
    if not content_type.startswith("application/pdf"):
        raise HTTPException(status_code=502, detail="Upstream pathway content was not a PDF.")

    response_headers = {"Content-Disposition": "inline"}
    for header_name in (
        "Accept-Ranges",
        "Content-Length",
        "Content-Range",
        "Cache-Control",
        "ETag",
        "Last-Modified",
    ):
        header_value = upstream_response.headers.get(header_name)
        if header_value:
            response_headers[header_name] = header_value

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type="application/pdf",
        headers=response_headers,
    )


# Public chat endpoint
@app.post("/chat/public", response_model=ChatResponse)
async def chat_public(request: ChatRequest):
    session_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        top_k = request.top_k or 5
        selected_pathway_doc_name = resolve_pathway_doc_name(request.pathway_id)

        print("="*60)
        print("=== NEW QUERY RECEIVED ===")
        print(f"Session ID: {session_id}")
        print(f"Query: {request.query}")
        print("Provider: gemini")
        print(f"Model: {request.model_name}")
        print(f"Top K: {request.top_k}")

        db_handle = get_supabase_client()

        print("\n[1/6] Generating embeddings...")
        query_emb = get_embeddings([request.query], model_key=request.embedding_model, is_query=True)[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        print(f"✓ Embedding generated: dimension={len(query_emb_list)}\n")

        # Retrieve top-k chunks
        print(f"\n[2/6] Retrieving top {top_k} chunks...")
        results = retrieve_chunks_by_model(db_handle, request.query, top_k=top_k, pathway_id=request.pathway_id, model_key=request.embedding_model)
        print(f"DEBUG: Retrieved {len(results)} results\n")

        citations = []
        for r in results:
            citation = {
                "chunk_id": int(r["chunk_id"]),
                "chunk_text": str(r["chunk_text"]),
                "chunk_length": int(r["chunk_length"]) if r["chunk_length"] is not None else 0,
                "source_file": str(r.get("source_file") or r.get("pathway_id") or ""),
                "similarity_score": float(r.get("distance", r.get("similarity", 0.0))),
            }
            citations.append(citation)

        if citations:
            print("\nTop chunk:")
            print(f"  - Source: {citations[0]['source_file']}")
            print(f"  - Similarity: {citations[0]['similarity_score']:.4f}")
        
        print(f"\n[3/6] Sending to {request.model}...")
        # Generate response using LLM
        if request.model == "gemini":
            # Use Gemini API (default)
            response_text = original_rag_api_llm(db_handle, request.query, top_k=top_k, 
                                       model_name=request.model_name, api_provider="gemini",
                                       pathway_id=request.pathway_id,
                                       retrieved_results=results)
        else:
            # Default to Gemini if unknown model specified
            response_text = original_rag_api_llm(db_handle, request.query, top_k=top_k, 
                                       model_name="gemini-2.5-flash", api_provider="gemini",
                                       pathway_id=request.pathway_id,
                                       retrieved_results=results)
        
        print(f"\n[4/6] Response received:")
        print(f"✓ Length: {len(response_text)} chars")
        print(f"✓ Preview: {response_text[:150]}...")

        response_time_ms = int((time.time() - start_time) * 1000)

        print("\n[5/6] Logging to Supabase...")
        query_logger.log_query(
            session_id=session_id,
            user_query=request.query,
            query_embedding=query_emb_list,
            bot_response=response_text,
            retrieved_chunks=citations,
            llm_provider="gemini",
            llm_model=request.model_name or "gemini-2.5-flash",
            response_time_ms=response_time_ms,
            pathway_id=request.pathway_id,
            user_role="public",
        )

        print("\n[6/6] Creating response...")

        client_citations = [
            {
                "chunk_id": str(c["chunk_id"]),
                "chunk_text": c["chunk_text"],
                "chunk_length": c["chunk_length"],
                "source_file": c["source_file"],
            }
            for c in citations
        ]

        response_obj = ChatResponse(
            response=response_text,
            citations=client_citations,
            timestamp=datetime.now().isoformat(),
            role="public",
        )

        print(f"✓ Response time: {response_time_ms}ms")
        print("=" * 60)
        print("=== QUERY COMPLETE ===")
        print("=" * 60)

        return response_obj

    except Exception as e:
        print("\n" + "!" * 60)
        print("!!! ERROR OCCURRED !!!")
        print(f"Query: {request.query}")
        print(f"Error: {e}")
        print("!" * 60)
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Practitioner chat endpoint
# @app.post("/chat/practitioner", response_model=ChatResponse)
# async def chat_practitioner(
#     request: PractitionerChatRequest,
#     user_info: Dict[str, Any] = Depends(verify_token)
# ):
#     """
#     Practitioner chat endpoint - requires authentication.
#     Includes memory/context from previous interactions.
#     """
#     if not user_info or user_info.get("role") != "practitioner":
#         raise HTTPException(status_code=403, detail="Access denied. Practitioner role required.")
    
#     user_id = user_info.get("user_id")
#     session_id = str(uuid.uuid4())
#     start_time = time.time()
    
#     try:
#         db_handle = get_supabase_client()
#         history_response = (
#             db_handle.table(PATHWAY_QUERIES_TABLE)
#             .select("user_query, bot_response, created_at")
#             .eq("user_id", user_id)
#             .eq("user_role", "practitioner")
#             .order("created_at", desc=True)
#             .limit(5)
#             .execute()
#         )
#         history = history_response.data or []
        
#         # Build context from history
#         context_history = "\n\nPrevious conversation:\n"
#         for h in reversed(history):
#             response_preview = (h.get('bot_response') or "")[:200]
#             context_history += f"Q: {h.get('user_query', '')}\nA: {response_preview}...\n\n"
        
#         # Perform RAG
#         query_emb = get_embeddings([request.query])[0]
#         query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        
#         results = retrieve_chunks(db_handle, query_emb_list, top_k=request.top_k, pathway_id=request.pathway_id)
#         citations = [
#             {
#                 "chunk_id": str(r.get("chunk_id", "")),
#                 "chunk_text": str(r.get("chunk_text", "")),
#                 "chunk_length": int(r.get("chunk_length") or 0),
#                 "source_file": str(r.get("source_file") or r.get("pathway_id") or "")
#             }
#             for r in results
#         ]
        
#         # Generate response with context using specified model
#         if request.model == "ollama":
#             response_text = rag_ollama(db_handle, request.query, top_k=request.top_k,
#                                        pathway_id=request.pathway_id, retrieved_results=results)
#         elif request.model == "gemini":
#             response_text = rag_api_llm(db_handle, request.query, top_k=request.top_k, 
#                                        model_name=request.model_name, api_provider="gemini",
#                                        pathway_id=request.pathway_id, retrieved_results=results)
#         else:
#             # Default to Gemini
#             response_text = rag_api_llm(db_handle, request.query, top_k=request.top_k, 
#                                        model_name="gemini-2.5-flash", api_provider="gemini",
#                                        pathway_id=request.pathway_id, retrieved_results=results)

#         response_time_ms = int((time.time() - start_time) * 1000)
#         query_logger.log_query(
#             session_id=session_id,
#             user_query=request.query,
#             query_embedding=query_emb_list,
#             bot_response=response_text,
#             retrieved_chunks=citations,
#             llm_provider=request.model,
#             llm_model=request.model_name if request.model == "gemini" else "llama2",
#             response_time_ms=response_time_ms,
#             pathway_id=request.pathway_id,
#             user_id=user_id,
#             user_role="practitioner"
#         )
        
#         return ChatResponse(
#             response=response_text,
#             citations=citations,
#             timestamp=datetime.now().isoformat(),
#             role="practitioner"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# Get practitioner history
@app.get("/history/{user_id}", response_model=List[ChatHistoryItem])
async def get_history(
    user_id: str,
    user_info: Dict[str, Any] = Depends(verify_token),
):
    """Get chat history for a practitioner."""
    if not user_info or (user_info.get("user_id") != user_id and user_info.get("role") not in ["admin", "hr"]):
        raise HTTPException(status_code=403, detail="Access denied.")

    try:
        db_handle = get_supabase_client()
        response = (
            db_handle.table(PATHWAY_QUERIES_TABLE)
            .select("user_query, bot_response, created_at")
            .eq("user_id", user_id)
            .eq("user_role", "practitioner")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        rows = response.data or []
        return [
            ChatHistoryItem(
                query=str(r.get("user_query", "")),
                response=str(r.get("bot_response", "")),
                timestamp=str(r.get("created_at", datetime.now().isoformat())),
            )
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
