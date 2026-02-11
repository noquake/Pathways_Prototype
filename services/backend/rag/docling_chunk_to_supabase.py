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
from pathlib import Path


MAX_TOKENS = 385

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
    # preferred = "*.doctags.txt"
    preferred = "*.md"
    md_files = list(md_dir.glob(preferred))
    print(f"Found {len(md_files)} files matching {preferred}")
    

    assert md_files, f"No markdown files found in md_dir"
    global_idx = 0
    for i, file in enumerate(md_files, 1):
        doc = DocumentConverter().convert(source=file).document
        doc_chunk_idx = 0
        for raw_chunk in chunker.chunk(dl_doc=doc):
            global_idx += 1
            doc_chunk_idx += 1
            contextualized_chunk = chunker.contextualize(raw_chunk)
            chunk_text = contextualized_chunk
            chunk_hash = hash_chunk_text(chunk_text)
            yield {
                "global_index": global_idx,
                "doc_name": file.stem,
                "doc_chunk_index": doc_chunk_idx,
                "chunk_hash": chunk_hash,
                "contextualized_chunk": contextualized_chunk,
                "chunk_text": chunk_text
            }


def create_db_connection():
    # connect to the PostgreSQL database, casting the connection from variable -> to a vector type for psycopg1
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)

    # creates a cursor object to start executing SQL commands
    cur = conn.cursor()

    cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')

    register_vector(conn)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            chunk_id BIGSERIAL PRIMARY KEY,
            chunk_hash TEXT UNIQUE NOT NULL,
            doc_name TEXT NOT NULL,
            doc_chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_length INTEGER NOT NULL,
            embedding vector(384) NOT NULL
        );
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS items_embedding_hnsw
        ON items
        USING hnsw (embedding vector_l2_ops);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS items_doc_name_idx
        ON items (doc_name);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS items_chunk_hash_idx
        ON items (chunk_hash);
    """)

    """ Index types for pgvector """
    # Approximate Nearest Neighbor (ANN) indexes for faster search MEDIUM to LARGE datasets, bad space
    cur.execute('CREATE INDEX IF NOT EXISTS embeddings ON items USING hnsw (embedding vector_l1_ops)')
    # Cluster-based ANN 
    # cur.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l1_ops) WITH (lists = 100)')
    
    return conn, cur

def get_embedding(contextualized_chunk):
    return model.encode(contextualized_chunk)



def insert_chunk_and_embedding_to_db(chunk, embedding, cur, conn):
    cur.execute(
        '''
        INSERT INTO items (
            chunk_hash,
            doc_name,
            doc_chunk_index,
            chunk_text,
            chunk_length,
            embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_hash) DO NOTHING
        ''',
        (
            chunk["chunk_hash"],
            chunk["doc_name"],
            chunk["doc_chunk_index"],
            chunk["contextualized_chunk"],
            len(chunk["contextualized_chunk"]),
            embedding.tolist() if hasattr(embedding, "tolist") else embedding,
        ),
    )
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
    # Add more as you process files
}

def get_pathway_category(pathway_id: str) -> str:
    """Get category from lookup table or return 'uncategorized'."""
    return PATHWAY_CATEGORIES.get(pathway_id, "uncategorized")

def main():
    print("Creating connection to PGVector database...\n")
    conn, cur = create_db_connection()

    print(
        "Starting for loop to stream chunk generation -> "
        "embedding -> DB insertion...\n"
    )

    # Support both Docker and local paths
    md_dir = os.getenv("TRANSFORMED_FILES_DIR", "/app/data/transformed_files")
    print(f"Looking for transformed files in: {md_dir}\n")
    
    for item in generate_chunks(md_dir, chunker):
        print(f"Processing chunk #{item['global_index']}...\n")

        # contextualization already happened in the generator
        emb = get_embedding(item["contextualized_chunk"])
        print(f"Chunk #{item['global_index']} embedded. Now inserting into DB...\n")

        insert_chunk_and_embedding_to_db(item, emb, cur, conn)
        print(f"Chunk #{item['global_index']} inserted into DB.\n")

    conn.close()
    print("Chunking and embedding complete.\n")


if __name__ == "__main__":
    main()

# """ QUERY EXAMPLE """
# query = "What is Pathways?"
# query_emb = model.encode(query)
# cur.execute('''
#     SELECT chunk_index, chunk_text, chunk_length, source_file 
#     FROM items 
#     ORDER BY embedding <-> %s 
#     LIMIT 5
# ''', (query_emb,))
# results = cur.fetchall()
# for r in results:
#     chunk_index, chunk_text, chunk_length, source_file = r
#     print(f"Chunk Index: {chunk_index}")
#     print(f"Source File: {source_file}")
#     print(f"Chunk Length: {chunk_length}")
#     print(f"Chunk Text: {chunk_text[:200]}..." if len(chunk_text) > 200 else f"Chunk Text: {chunk_text}")
#     print("-" * 80)