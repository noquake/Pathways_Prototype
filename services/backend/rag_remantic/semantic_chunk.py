from pathlib import Path
import hashlib
import math
from sentence_transformers import SentenceTransformer # type: ignore
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import os
import re
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

MAX_TOKENS = 384

model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)

chunker = HybridChunker(
    tokenizer = tokenizer,
    merge_peers = True,
    overlap_tokens = math.ceil(MAX_TOKENS * 0.2), # Overlap by 20 percent
    respect_section_boundaries = True,
    max_table_cell_tokens = 384,  # preserve table rows
)

def hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def resolve_markdown_files(md_dir: Path):
    """Find markdown-like transformed files in the provided directory."""
    # Docling converter supports markdown; doctags text files are not accepted input.
    matches = list(md_dir.rglob("*.md"))

    # De-dupe while preserving stable order.
    unique_files = sorted(set(matches), key=lambda p: str(p))
    return unique_files

def extract_chunk_metadata(raw_chunk, doc_metadata):
    """
    Extract rich structural metadata from Docling chunk.
    
    Returns metadata about where this chunk came from in the document.
    """
    metadata = {
        "pathway_id": doc_metadata["pathway_id"],
        "doc_name": doc_metadata["doc_name"],
    }
    
    # 1. Page information (from provenance)
    if hasattr(raw_chunk, 'prov') and raw_chunk.prov:
        # Docling stores page numbers in prov
        pages = [p.page_no for p in raw_chunk.prov if hasattr(p, 'page_no')]
        metadata["page_numbers"] = list(set(pages)) if pages else []
        metadata["primary_page"] = pages[0] if pages else None
    
    # 2. Section/heading information
    if hasattr(raw_chunk, 'meta') and raw_chunk.meta:
        # Try to extract section heading
        if 'heading' in raw_chunk.meta:
            metadata["section_title"] = raw_chunk.meta['heading']
        if 'headings' in raw_chunk.meta:
            # Sometimes multiple hierarchical headings
            metadata["section_hierarchy"] = raw_chunk.meta['headings']
    
    # 3. Content type (table, paragraph, list, etc.)
    if hasattr(raw_chunk, 'self_ref') and hasattr(raw_chunk.self_ref, 'content_type'):
        metadata["chunk_type"] = raw_chunk.self_ref.content_type
    else:
        metadata["chunk_type"] = "text"  # Default
    
    # 4. Parent section (for finding related chunks)
    if hasattr(raw_chunk, 'path'):
        # Path shows document hierarchy like: /body/section[2]/paragraph[1]
        metadata["document_path"] = str(raw_chunk.path)
        # Extract parent section ID from path
        metadata["parent_section_id"] = extract_parent_section_from_path(raw_chunk.path)
    
    return metadata

def extract_parent_section_from_path(path_str):
    """
    Extract parent section identifier from document path.
    Example: '/body/section[2]/paragraph[1]' → 'section-2'
    """
    if not path_str:
        return None
    
    # Find the section in the path
    match = re.search(r'/section\[(\d+)\]', str(path_str))
    if match:
        return f"section-{match.group(1)}"
    
    # Fallback: use the entire path as ID
    return hashlib.md5(str(path_str).encode()).hexdigest()[:8]

# Retrieve docling-ized md files, generate chunks and append them to an all_chunks list
def generate_chunks(md_dir: str, chunker: HybridChunker):
    md_dir = Path(md_dir)
    md_files = resolve_markdown_files(md_dir)
    global_idx = 0
    for file in md_files:
        doc_metadata = extract_pathway_metadata(file.stem, file)
        doc_metadata["doc_file_path"] = str(file)
        
        yield {
            "type": "document",
            "metadata": doc_metadata
        }
        
        doc = DocumentConverter().convert(source=file).document
        doc_chunk_idx = 0
        
        # Track current section for sequence numbering
        current_section = None
        section_chunk_count = 0
        
        for raw_chunk in chunker.chunk(dl_doc=doc):
            global_idx += 1
            doc_chunk_idx += 1
            
            # Extract metadata BEFORE contextualizing
            chunk_metadata = extract_chunk_metadata(raw_chunk, doc_metadata)
            
            # Track sequence within section
            if chunk_metadata.get("parent_section_id") != current_section:
                current_section = chunk_metadata.get("parent_section_id")
                section_chunk_count = 1
            else:
                section_chunk_count += 1
            
            chunk_metadata["sequence_in_section"] = section_chunk_count
            
            # Now contextualize
            contextualized_chunk = chunker.contextualize(raw_chunk)
            chunk_text = contextualized_chunk
            chunk_hash = hash_chunk_text(chunk_text)
            
            yield {
                "type": "chunk",
                "global_index": global_idx,
                "pathway_id": doc_metadata["pathway_id"],
                "doc_chunk_index": doc_chunk_idx,
                "chunk_hash": chunk_hash,
                "chunk_text": chunk_text,
                "metadata": chunk_metadata  # ← ADD THIS
            }

def create_supabase_client():
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")
    supabase_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY or SUPABASE_PUBLISHABLE_KEY

    if not SUPABASE_URL or not supabase_key:
        raise ValueError(
            "Missing SUPABASE_URL or Supabase key. "
            "Set SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY / SUPABASE_PUBLISHABLE_KEY."
        )
    
    supabase: Client = create_client(SUPABASE_URL, supabase_key)
    return supabase

def create_tables():
    """
    Create tables in Supabase using SQL.
    Run this ONCE in Supabase SQL Editor, then you can comment it out.
    """
    
    sql = """
    -- Enable vector extension (if not already enabled)
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- pathway_documents table
    CREATE TABLE IF NOT EXISTS pathway_documents (
        pathway_id VARCHAR(100) PRIMARY KEY,
        doc_name TEXT UNIQUE NOT NULL,
        doc_display_name VARCHAR(255) NOT NULL,
        doc_version VARCHAR(50),
        doc_category VARCHAR(100),
        doc_file_path TEXT,
        doc_last_modified TIMESTAMPTZ,
        active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- pathway_chunks table
    CREATE TABLE IF NOT EXISTS pathway_chunks (
        chunk_id BIGSERIAL PRIMARY KEY,
        chunk_hash TEXT UNIQUE NOT NULL,
        pathway_id VARCHAR(100) NOT NULL REFERENCES pathway_documents(pathway_id),
        doc_chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_length INTEGER NOT NULL,
        embedding VECTOR(384) NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    -- Basic indexes
    CREATE INDEX IF NOT EXISTS idx_pathway_chunks_pathway_id
    ON pathway_chunks (pathway_id);
    
    CREATE INDEX IF NOT EXISTS idx_pathway_chunks_hash
    ON pathway_chunks (chunk_hash);
    """
    
    print("Copy and run this SQL in Supabase SQL Editor:")
    print("="*60)
    print(sql)
    print("="*60)

def get_embedding(contextualized_chunk):
    return model.encode(contextualized_chunk)

def insert_pathway_document(metadata: dict, supabase):
    """Insert or update pathway document metadata."""
    data = {
        "pathway_id": metadata["pathway_id"],
        "doc_name": metadata["doc_name"],
        "doc_display_name": metadata["doc_display_name"],
        "doc_version": metadata["doc_version"],
        "doc_category": get_pathway_category(metadata["pathway_id"]),
        "doc_file_path": metadata.get("doc_file_path"),
        "doc_last_modified": metadata["doc_last_modified"].isoformat()
    }
    supabase.table("semantic_pathway_documents").upsert(data).execute()

def insert_chunk_and_embedding_to_db(chunk, embedding, supabase):
    data = {
        "chunk_hash": chunk["chunk_hash"],
        "pathway_id": chunk["pathway_id"],
        "doc_chunk_index": chunk["doc_chunk_index"],
        "chunk_text": chunk["chunk_text"],
        "chunk_length": len(chunk["chunk_text"]),
        "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
        "metadata": chunk.get("metadata", {})  # ← ADD THIS
    }
    
    supabase.table("semantic_pathway_chunks").upsert(data).execute()

# ---------------------------------------------------- NEW DOC ----------------------------------------------------

def extract_pathway_metadata(doc_name: str, file_path: Path) -> dict:
    """
    Extract metadata from document filename.
    
    Examples:
        "anaphylaxis_-_1.16.25" → pathway_id="anaphylaxis", version="1.16.25"
        "status_epilepticus_module_-_9.27.23_pdf" → pathway_id="status-epilepticus", version="9.27.23"
    """
    
    # Remove file extension and cleanup
    clean_name = doc_name.replace("_pdf", "").replace(".md", "")
    
    # Extract version (pattern: numbers.numbers.numbers or date-like)
    version_match = re.search(r'(\d+\.\d+\.\d+)', clean_name)
    doc_version = version_match.group(1) if version_match else None
    
    # Remove version and separators to get pathway name
    pathway_name = re.sub(r'_-_\d+\.\d+\.\d+', '', clean_name)
    pathway_name = re.sub(r'[-_]+', '-', pathway_name)  # Replace _ and - with single -
    pathway_name = pathway_name.strip('-').lower()
    
    # Get file modification time. Fallback to "now" for synthetic test paths.
    if file_path.exists():
        doc_last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
    else:
        doc_last_modified = datetime.now()
    
    # Create display name (capitalize and clean up)
    display_name = pathway_name.replace('-', ' ').title()
    
    return {
        "pathway_id": pathway_name,
        "doc_name": doc_name,
        "doc_display_name": display_name,
        "doc_version": doc_version,
        "doc_last_modified": doc_last_modified
    }

# Category mapping (manual - you'll need to maintain this)
PATHWAY_CATEGORIES = {
    "anaphylaxis": "emergency",
    "dka": "endocrine",
    "sepsis": "infectious-disease",
    "status-epilepticus": "neurology",
    "animal-bite": "infectious-disease",
    # TODO: ADD MORE / REFINE BASED ON TOPICS
}

def test_extraction():
    """Test metadata extraction on sample filenames."""
    test_files = [
        "anaphylaxis_-_1.16.25.md",
        "status_epilepticus_module_-_9.27.23_pdf.md",
        "animal-and-human-bite-and-soft-tissue-infection-algorithm-8.12.25.md"
    ]
    
    print("Testing metadata extraction:\n")
    for filename in test_files:
        stem = filename.replace(".md", "")
        meta = extract_pathway_metadata(stem, Path(f"/fake/{filename}"))
        print(f"File: {filename}")
        print(f"  pathway_id: {meta['pathway_id']}")
        print(f"  display_name: {meta['doc_display_name']}")
        print(f"  version: {meta['doc_version']}")
        print(f"  category: {get_pathway_category(meta['pathway_id'])}")
        print()

def get_pathway_category(pathway_id: str) -> str:
    """Get category from lookup table or return 'uncategorized'."""
    return PATHWAY_CATEGORIES.get(pathway_id, "uncategorized")

def main():
    print("="*60)
    print("PATHWAY INGESTION TO SUPABASE")
    print("="*60 + "\n")
    
    print("Creating connection to database...\n")
    supabase = create_supabase_client()  # Changed this line
        
    md_dir = os.getenv("TRANSFORMED_FILES_DIR", "/app/data/transformed_files")
    print(f"Looking for files in: {md_dir}\n")
    
    chunk_count = 0
    doc_count = 0
    error_count = 0
    
    for item in generate_chunks(md_dir, chunker):
        try:
            if item["type"] == "document":
                print(f"📄 Registering pathway: {item['metadata']['pathway_id']}")
                insert_pathway_document(item["metadata"], supabase)
                doc_count += 1
            
            elif item["type"] == "chunk":
                print(f"  └─ Chunk #{item['global_index']}...", end=" ")
                emb = get_embedding(item["chunk_text"])
                insert_chunk_and_embedding_to_db(item, emb, supabase)
                chunk_count += 1
                print("✓")
        
        except Exception as e:
            error_count += 1
            print(f"\n❌ Error processing item: {e}")
            import traceback
            traceback.print_exc()  # This will help you debug
            continue
    
    # Summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Documents processed: {doc_count}")
    print(f"Chunks created: {chunk_count}")
    print(f"Errors: {error_count}")
    print("="*60 + "\n")
    
    # Create HNSW index AFTER data is loaded
    if chunk_count > 0:
        print(f"Creating vector index on {chunk_count} chunks...")
        print("⚠️  Run this SQL in Supabase SQL Editor:")
        print("""
        CREATE INDEX IF NOT EXISTS idx_pathway_chunks_embedding
        ON pathway_chunks USING hnsw (embedding vector_cosine_ops);
        """)
        print("(You only need to do this once)")
    else:
        print("⚠️  No chunks to index.\n")
    
    # No conn.close() needed
    print("Chunking and embedding complete.\n")


if __name__ == "__main__":
    main()
