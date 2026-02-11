from pathlib import Path
import psycopg2
from pgvector.psycopg2 import register_vector
import hashlib
from sentence_transformers import SentenceTransformer # type: ignore
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import os
import re
from datetime import datetime

MAX_TOKENS = 384

model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)

chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True,
)

def hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Retrieve docling-ized md files, generate chunks and append them to an all_chunks list
def generate_chunks(md_dir: str, chunker: HybridChunker):
    md_dir = Path(md_dir)
    md_files = list(md_dir.glob("*.md"))
    
    print(f"Found {len(md_files)} markdown files")
    
    global_idx = 0
    for file in md_files:
        # Extract document metadata
        doc_metadata = extract_pathway_metadata(file.stem, file)
        doc_metadata["doc_file_path"] = str(file)
        
        # Yield document metadata first (to be inserted before chunks)
        yield {
            "type": "document",
            "metadata": doc_metadata
        }
        
        # Then process chunks
        doc = DocumentConverter().convert(source=file).document
        doc_chunk_idx = 0
        
        for raw_chunk in chunker.chunk(dl_doc=doc):
            global_idx += 1
            doc_chunk_idx += 1
            contextualized_chunk = chunker.contextualize(raw_chunk)
            chunk_text = contextualized_chunk
            chunk_hash = hash_chunk_text(chunk_text)
            
            yield {
                "type": "chunk",
                "global_index": global_idx,
                "pathway_id": doc_metadata["pathway_id"],  # Use extracted pathway_id
                "doc_chunk_index": doc_chunk_idx,
                "chunk_hash": chunk_hash,
                "chunk_text": chunk_text
            }

def create_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    register_vector(conn)
    
    # pathway_documents table
    cur.execute("""
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
    """)
    
    # pathway_chunks table
    cur.execute("""
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
    """)
    
    # Basic indexes only (HNSW created after data load)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pathway_chunks_pathway_id
        ON pathway_chunks (pathway_id);
    """)
    
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pathway_chunks_hash
        ON pathway_chunks (chunk_hash);
    """)
    
    conn.commit()
    return conn, cur

def get_embedding(contextualized_chunk):
    return model.encode(contextualized_chunk)

def insert_pathway_document(metadata: dict, cur, conn):
    """Insert or update pathway document metadata."""
    cur.execute("""
        INSERT INTO pathway_documents (
            pathway_id,
            doc_name,
            doc_display_name,
            doc_version,
            doc_category,
            doc_file_path,
            doc_last_modified
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (pathway_id) DO UPDATE SET
            doc_name = EXCLUDED.doc_name,
            doc_version = EXCLUDED.doc_version,
            doc_last_modified = EXCLUDED.doc_last_modified
    """, (
        metadata["pathway_id"],
        metadata["doc_name"],
        metadata["doc_display_name"],
        metadata["doc_version"],
        get_pathway_category(metadata["pathway_id"]),
        metadata.get("doc_file_path"),
        metadata["doc_last_modified"]
    ))
    conn.commit()

def insert_chunk_and_embedding_to_db(chunk, embedding, cur, conn):
    cur.execute("""
        INSERT INTO pathway_chunks (
            chunk_hash,
            pathway_id,
            doc_chunk_index,
            chunk_text,
            chunk_length,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_hash) DO NOTHING
    """, (
        chunk["chunk_hash"],
        chunk["pathway_id"],  # Now using pathway_id instead of doc_name
        chunk["doc_chunk_index"],
        chunk["chunk_text"],
        len(chunk["chunk_text"]),
        embedding.tolist() if hasattr(embedding, "tolist") else embedding,
    ))
    conn.commit()


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
    
    # Get file modification time
    doc_last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
    
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
    conn, cur = create_db_connection()
    
    md_dir = os.getenv("TRANSFORMED_FILES_DIR", "/app/data/transformed_files")
    print(f"Looking for files in: {md_dir}\n")
    
    chunk_count = 0
    doc_count = 0
    error_count = 0
    
    for item in generate_chunks(md_dir, chunker):
        try:
            if item["type"] == "document":
                print(f"📄 Registering pathway: {item['metadata']['pathway_id']}")
                insert_pathway_document(item["metadata"], cur, conn)
                doc_count += 1
            
            elif item["type"] == "chunk":
                print(f"  └─ Chunk #{item['global_index']}...", end=" ")
                emb = get_embedding(item["chunk_text"])
                insert_chunk_and_embedding_to_db(item, emb, cur, conn)
                chunk_count += 1  # ← CRITICAL: Increment counter
                print("✓")
        
        except Exception as e:
            error_count += 1
            print(f"❌ Error: {e}")
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
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_pathway_chunks_embedding
                ON pathway_chunks USING hnsw (embedding vector_cosine_ops);
            """)
            conn.commit()
            print("✓ Vector index created successfully\n")
        except Exception as e:
            print(f"⚠ Warning: Could not create HNSW index: {e}")
            print("You can create it manually later.\n")
    else:
        print("⚠ No chunks to index.\n")
    
    conn.close()
    print("Chunking and embedding complete.\n")


if __name__ == "__main__":
    main()

