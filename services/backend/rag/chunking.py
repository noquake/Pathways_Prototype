from pathlib import Path
import hashlib
import math
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import os
import re
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
from .embeddings import get_embeddings
from .models import EMBEDDING_MODELS, DEFAULT_MODEL

load_dotenv()

def hash_chunk_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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

def get_pathway_category(pathway_id: str) -> str:
    """Get category from lookup table or return 'uncategorized'."""
    return PATHWAY_CATEGORIES.get(pathway_id, "uncategorized")

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

def generate_chunks_with_model(model_key: str = DEFAULT_MODEL):

    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    config = EMBEDDING_MODELS[model_key]
    model_name = config["model_name"]
    dimension = config["dimension"]
    table_name = config["table"]

    print(f"="*60)
    print(f"PATHWAY CHUNKING WITH {model_key.upper()}")
    print(f"Model: {model_name}")
    print(f"Dimension: {dimension}")
    print(f"Table: {table_name}")
    print(f"="*60 + "\n")

    tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    max_tokens=dimension,
    )

    chunker = HybridChunker(
    tokenizer = tokenizer,
    merge_peers = True,
    overlap_tokens = math.ceil(dimension * 0.2), # Overlap by 20 percent
    respect_section_boundaries = True,
    max_table_cell_tokens = dimension,  # preserve table rows
    )

    supabase = create_supabase_client()
    
    md_dir = Path(os.getenv("TRANSFORMED_FILES_DIR", "app/data/transformed_files"))
    md_files = list(md_dir.glob("*.md"))

    print(f"Found {len(md_files)} markdown files\n")

    chunk_count = 0
    doc_count = 0
    error_count = 0

    for file in md_files:
        try:
            # Extract document metadata
            doc_metadata = extract_pathway_metadata(file.stem, file)
            doc_metadata["doc_file_path"] = str(file)
            
            # Insert document
            print(f"📄 Processing: {doc_metadata['pathway_id']}")
            supabase.table("pathway_documents").upsert({
                "pathway_id": doc_metadata["pathway_id"],
                "doc_name": doc_metadata["doc_name"],
                "doc_display_name": doc_metadata["doc_display_name"],
                "doc_version": doc_metadata["doc_version"],
                "doc_category": get_pathway_category(doc_metadata["pathway_id"]),
                "doc_file_path": doc_metadata.get("doc_file_path"),
                "doc_last_modified": doc_metadata["doc_last_modified"].isoformat()
            }).execute()
            doc_count += 1
            
            # Process chunks
            doc = DocumentConverter().convert(source=file).document
            doc_chunk_idx = 0
            
            for raw_chunk in chunker.chunk(dl_doc=doc):
                doc_chunk_idx += 1
                chunk_text = chunker.contextualize(raw_chunk)
                chunk_hash = hash_chunk_text(chunk_text)
                
                # Generate embedding
                embedding = get_embeddings([chunk_text], model_key=model_key)[0]
                
                # Insert chunk
                supabase.table("pathways_chunks_mpnet").upsert({
                    "chunk_hash": chunk_hash,
                    "pathway_id": doc_metadata["pathway_id"],
                    "doc_chunk_index": doc_chunk_idx,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text),
                    "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                    "metadata": {}  # Add metadata extraction if needed
                }).execute()
                
                chunk_count += 1
                if chunk_count % 10 == 0:
                    print(f"  └─ {chunk_count} chunks processed...")

        except Exception as e:
            error_count += 1
            print(f"❌ Error: {e}")
            continue

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents: {doc_count}")
    print(f"Chunks: {chunk_count}")
    print(f"Errors: {error_count}")
    print(f"="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    model_key = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    generate_chunks_with_model(model_key)


