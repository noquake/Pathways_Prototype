from pathlib import Path
from sentence_transformers import SentenceTransformer # type: ignore
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_markdown(md_texts: str, source_file: str = None, max_len: int = 1000) -> List[Dict[str, Any]]:
    """
    Chunk markdown text and return chunks with metadata.
    
    Args:
        md_texts: The markdown text to chunk
        source_file: The source file name/path (optional)
        max_len: Maximum length for each chunk
    
    Returns:
        List of dictionaries, each containing chunk text and metadata
    """
    chunks = []
    current = []
    chunk_index = 0

    for line in md_texts.split("\n"):
        if line.startswith("# "):
            if current:
                chunk_text = "\n".join(current)
                chunks.append({
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_length": len(chunk_text),
                    "source_file": source_file,
                })
                chunk_index += 1
                current = []
        current.append(line)
        if sum(len(l) for l in current) + len(line) > max_len:
            chunk_text = "\n".join(current)
            chunks.append({
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "chunk_length": len(chunk_text),
                "source_file": source_file,
            })
            chunk_index += 1
            current = []
    if current:
        chunk_text = "\n".join(current)
        chunks.append({
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": len(chunk_text),
            "source_file": source_file,
        })
    return chunks

all_chunks = []
for md_path in Path("scratch").glob("*.md"):
    md_texts = md_path.read_text()
    chunks = chunk_markdown(md_texts, source_file=md_path.name)
    all_chunks.extend(chunks)
    print(md_path.name, len(chunks))

chunks = all_chunks  # Keep for backward compatibility with rest of code


# """ Testing to see if the output actually comes out as expected """
# output_path = Path("scratch/chunks.txt")
# with output_path.open("w") as f:
#     for i, chunk in enumerate(chunks):
#         f.write(f"\n\n===== CHUNK {i} =====\n\n")
#         f.write(chunk)
#         f.write("\n")

def embed_chunks(chunks):
    # Extract chunk texts if chunks are dictionaries
    if chunks and isinstance(chunks[0], dict):
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    else:
        chunk_texts = chunks
    
    embeddings = model.encode(chunk_texts)
    with open("embeddings.txt", "w") as f:
        for i, emb in enumerate(embeddings):
            chunk_info = f"Chunk {i}"
            if isinstance(chunks[i], dict):
                chunk_info = f"Chunk {i} (Source: {chunks[i].get('source_file', 'N/A')})"
            f.write(f"{chunk_info}:\n{emb.tolist() if hasattr(emb, 'tolist') else emb}\n\n")
    
embed_chunks(chunks) # Run the embedding function

# connect to the PostgreSQL database, casting the connection from variable -> to a vector type for psycopg2
conn = psycopg2.connect("dbname=pathways user=admin password=password host=localhost port=5432")

# creates a cursor object to start executing SQL commands
cur = conn.cursor()
cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
register_vector(conn)

# create table of items with vector embeddings and metadata
cur.execute('''
    CREATE TABLE IF NOT EXISTS items (
        chunk_id bigserial PRIMARY KEY,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_length INTEGER NOT NULL,
        source_file VARCHAR(255),
        embedding vector(384) NOT NULL
    )
''')

# embed by chunk, and insert into the database
for chunk in chunks:
    # Extract chunk data if it's a dictionary
    if isinstance(chunk, dict):
        chunk_index = chunk["chunk_index"]
        chunk_text = chunk["chunk_text"]
        chunk_length = chunk["chunk_length"]
        source_file = chunk["source_file"]
    else:
        # Fallback for backward compatibility (shouldn't happen with new code)
        chunk_index = 0
        chunk_text = chunk
        chunk_length = len(chunk)
        source_file = None
    
    emb = model.encode(chunk_text)
    cur.execute('''
        INSERT INTO items (chunk_index, chunk_text, chunk_length, source_file, embedding) 
        VALUES (%s, %s, %s, %s, %s)
    ''', (chunk_index, chunk_text, chunk_length, source_file, emb))
    conn.commit()

""" Index types for pgvector """
# Approximate Nearest Neighbor (ANN) indexes for faster search MEDIUM to LARGE datasets, bad space
cur.execute('CREATE INDEX IF NOT EXISTS embeddings ON items USING hnsw (embedding vector_l2_ops)')
# Cluster-based ANN 
# cur.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)')



""" QUERY EXAMPLE """
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
