import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import os
from google import genai
# from ollama import Client

from rag.retrieval import retrieve_chunks
from rag.embeddings import get_embeddings

# API Keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CITATION_RE = re.compile(r"\[(\d+)\]")

# ----------------------
# Models and DB
# ----------------------
# ollama_client = Client(host="http://localhost:11434")  # default port for local Ollama server


def clean_query(query: str) -> str:
    """
    Fix typos and grammar in every query before retrieval.
    Uses Gemini flash so it's fast. Returns the original query on failure.
    Medical terminology is preserved — the model is instructed not to alter it.
    """
    if not GEMINI_API_KEY or not query.strip():
        return query

    prompt = (
        "Fix any spelling mistakes and grammar errors in the following clinical query. "
        "Do NOT change medical terms, drug names, or clinical abbreviations. "
        "Return ONLY the corrected query with no explanation or extra text.\n\n"
        f"Query: {query}\n\n"
        "Corrected query:"
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        cleaned = response.text.strip()
        return cleaned if cleaned else query
    except Exception as e:
        print(f"⚠ Query cleaning failed, using original: {e}")
        return query


def rewrite_query(query: str, conversation_history: list) -> str:
    """
    Use Gemini Flash to rewrite a follow-up query into a self-contained question
    based on conversation history, improving retrieval accuracy.
    """
    if not GEMINI_API_KEY or not conversation_history:
        return query

    history_text = "\n".join(
        f"{(turn.role if hasattr(turn, 'role') else turn['role']).capitalize()}: "
        f"{turn.content if hasattr(turn, 'content') else turn['content']}"
        for turn in conversation_history[-6:]
    )

    prompt = (
        "Given the following conversation history and a follow-up question, "
        "rewrite the follow-up question as a single, self-contained question "
        "that includes all necessary context from the conversation. "
        "Return ONLY the rewritten question with no explanation.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {query}\n\n"
        "Rewritten question:"
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        rewritten = response.text.strip()
        return rewritten if rewritten else query
    except Exception as e:
        print(f"⚠ Query rewriting failed, using original: {e}")
        return query


def build_context(results):
    """Build prompt context string from retrieval results."""
    if results and isinstance(results[0], dict):
        context_lines = []
        for i, r in enumerate(results):
            source_docs_field = r.get("source_docs") or []
            if isinstance(source_docs_field, list):
                source_doc = source_docs_field[0] if source_docs_field else None
            else:
                source_doc = str(source_docs_field).strip() or None
            source = source_doc or r.get("pathway_id") or "unknown"
            text = r.get("chunk_text", "")
            context_lines.append(f"[{i+1}] {source}: {text}")
        return "\n\n".join(context_lines)
    context_lines = []
    for i, r in enumerate(results):
        if isinstance(r, (tuple, list)) and len(r) >= 4:
            source = r[3]
            text = r[1]
        elif isinstance(r, (tuple, list)) and len(r) >= 2:
            source = r[1]
            text = r[0]
        else:
            source = "unknown"
            text = str(r)
        context_lines.append(f"[{i+1}] {source}: {text}")
    return "\n\n".join(context_lines)


def append_sources_legend(
    answer: str,
    chunk_document_numbers: Optional[List[Optional[int]]] = None,
    citation_documents: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Map chunk citations to document citations and append a PDF legend."""
    normalized_answer = (answer or "").strip()
    if not normalized_answer or not citation_documents or not chunk_document_numbers:
        return normalized_answer

    used_document_numbers = []
    seen_document_numbers = set()
    for match in CITATION_RE.finditer(normalized_answer):
        chunk_number = int(match.group(1))
        if not 1 <= chunk_number <= len(chunk_document_numbers):
            continue
        document_number = chunk_document_numbers[chunk_number - 1]
        if document_number is None or document_number in seen_document_numbers:
            continue
        used_document_numbers.append(document_number)
        seen_document_numbers.add(document_number)

    if not used_document_numbers:
        # LLM cited no chunks inline; still list all retrieved source docs
        legend_lines = ["Sources:"]
        for i, doc in enumerate(citation_documents, start=1):
            legend_lines.append(f"[{i}] {doc['pdf_name']}")
        return f"{normalized_answer}\n\n" + "\n".join(legend_lines)

    document_renumber_map = {
        original_document_number: replacement_number
        for replacement_number, original_document_number in enumerate(
            used_document_numbers,
            start=1,
        )
    }

    def replace_citation(match: re.Match[str]) -> str:
        chunk_number = int(match.group(1))
        if not 1 <= chunk_number <= len(chunk_document_numbers):
            return match.group(0)
        document_number = chunk_document_numbers[chunk_number - 1]
        if document_number is None:
            return match.group(0)
        replacement_number = document_renumber_map.get(document_number)
        return f"[{replacement_number}]" if replacement_number is not None else match.group(0)

    normalized_answer = CITATION_RE.sub(replace_citation, normalized_answer).strip()

    legend_lines = ["Sources:"]
    for replacement_number, original_number in enumerate(used_document_numbers, start=1):
        legend_lines.append(
            f"[{replacement_number}] {citation_documents[original_number - 1]['pdf_name']}"
        )

    # Append any retrieved docs the LLM didn't explicitly cite
    next_number = len(used_document_numbers) + 1
    for i in range(1, len(citation_documents) + 1):
        if i not in seen_document_numbers:
            legend_lines.append(f"[{next_number}] {citation_documents[i - 1]['pdf_name']}")
            next_number += 1

    return f"{normalized_answer}\n\n" + "\n".join(legend_lines)

# ----------------------
# Retrieval + RAG with API-based LLMs
# ----------------------
def rag_api_llm(
    supabase,
    query: str,
    top_k: int = 5,
    model_name: str = "gpt-4",
    api_provider: str = "gemini",
    pathway_id: str = None,
    retrieved_results=None,
    conversation_history: list = None,
    chunk_document_numbers: Optional[List[Optional[int]]] = None,
    citation_documents: Optional[List[Dict[str, str]]] = None,
):
    """
    Retrieve top-k chunks and use an API-based LLM (OpenAI, Gemini, etc.) to answer the query.
    
    Args:
        supabase: Supabase client
        query: User query string
        top_k: Number of top chunks to retrieve
        model_name: Name of the model to use
        api_provider: API provider to use ("openai" or "gemini")
    """
    results = retrieved_results
    if results is None:
        query_emb = get_embeddings([query])[0]
        query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
        results = retrieve_chunks(supabase, query_emb_list, top_k=top_k, pathway_id=pathway_id)

    if not results:
        print("No relevant chunks found.")
        return "No relevant chunks found in pathway documents."

    context = build_context(results)

    history_section = ""
    if conversation_history:
        turns = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation_history[-6:]
        )
        history_section = f"\nConversation so far:\n{turns}\n"

    prompt = f"""
You are a clinical assistant that STRICTLY follows institutional protocols.

CRITICAL INSTRUCTIONS:
- You MUST answer ONLY based on the provided context sources below
- The context sources are AUTHORITATIVE clinical protocols that you must follow exactly
- DO NOT use any external knowledge or add caveats about missing information
- DO NOT say "the context doesn't contain" - if information appears in the sources or their references, state it definitively
- When information is in the sources, present it as established medical protocol
- ALWAYS cite sources using [1], [2], etc. when referencing information
{history_section}
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
        answer = append_sources_legend(answer, chunk_document_numbers, citation_documents)
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
            "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
            "gemini-pro": "gemini-2.5-flash",  # Map old name to new model
            "gemini-pro-vision": "gemini-2.5-pro",
            "gemini-flash-latest": "gemini-flash-latest",
            "gemini-pro-latest": "gemini-pro-latest",
        }
        
        # Get the correct model name
        if model_name.startswith("gpt"):
            model_name = "gemini-3.1-flash-lite-preview"  # Default for OpenAI model names

        actual_model = model_mapping.get(model_name, "gemini-3.1-flash-lite-preview")
        
        # Generate response using new API
        response = client.models.generate_content(
            model=actual_model,
            contents=prompt
        )
        
        # Extract answer
        answer = append_sources_legend(response.text, chunk_document_numbers, citation_documents)
        print("\n=== Gemini Answer ===\n")
        print(answer)
        return answer
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}. Use 'openai' or 'gemini'.")

# ----------------------
# Retrieval + RAG with local LLaMA
# ----------------------
# def rag_ollama(supabase, query: str, top_k: int = 5, model_name: str = "llama2", pathway_id: str = None, retrieved_results=None):
#     """
#     Retrieve top-k chunks and use local LLaMA (Ollama) to answer the query.
#     """
#     results = retrieved_results
#     if results is None:
#         query_emb = get_embeddings([query])[0]
#         query_emb_list = query_emb.tolist() if hasattr(query_emb, "tolist") else query_emb
#         results = retrieve_chunks(supabase, query_emb_list, top_k=top_k, pathway_id=pathway_id)

#     if not results:
#         print("No relevant chunks found.")
#         return "No relevant chunks found in pathway documents."

#     context = build_context(results)
        
#     prompt = f"""
# You are a clinical assistant that STRICTLY follows institutional protocols.

# CRITICAL INSTRUCTIONS:
# - You MUST answer ONLY based on the provided context sources below
# - The context sources are AUTHORITATIVE clinical protocols that you must follow exactly
# - DO NOT use any external knowledge or add caveats about missing information
# - DO NOT say "the context doesn't contain" - if information appears in the sources or their references, state it definitively
# - When information is in the sources, present it as established medical protocol
# - ALWAYS cite sources using [1], [2], etc. when referencing information

# Context Sources (AUTHORITATIVE INSTITUTIONAL PROTOCOLS):
# {context}

# Question:
# {query}

# Provide a definitive answer based ONLY on the context above. Present the information from the sources as established protocol without hedging or adding external caveats.

# Answer:
# """

#     response = ollama_client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
#     answer = response['message']['content']
#     print("\n=== Local LLaMA Answer ===\n")
#     print(answer)
#     return answer

# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()
    supabase = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY"),
    )

    query = input("Enter your query: ")
    # Use Gemini by default (with gemini-3.1-flash-lite-preview for faster responses)
    rag_api_llm(supabase, query, top_k=5, model_name="gemini-3.1-flash-lite-preview", api_provider="gemini")
