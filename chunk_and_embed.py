from pathlib import Path
from sentence_transformers import SentenceTransformer # type: ignore
import psycopg2
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any

model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to create chunks with single file its metadata, stored in file_chunks
def chunk(texts: str, source_file: str = None, max_len: int = 1000) -> List[Dict[str, Any]]:
    """
    Chunk markdown text in a file and return chunks with metadata.
    
    Args:
        md_texts: The markdown text to chunk
        source_file: The source file name/path (optional)
        max_len: Maximum length for each chunk
    
    Returns:
        List of dictionaries, each containing chunk text and metadata
    """
    file_chunks = []
    current = []
    chunk_index = 0

    for line in texts.split("\n"):
        if line.startswith("# "):
            if current:
                chunk_text = "\n".join(current)
                file_chunks.append({
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
            file_chunks.append({
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "chunk_length": len(chunk_text),
                "source_file": source_file,
            })
            chunk_index += 1
            current = []
    if current:
        chunk_text = "\n".join(current)
        file_chunks.append({
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
            "chunk_length": len(chunk_text),
            "source_file": source_file,
        })
    return file_chunks

# Generating chunks from markdown files in the "scratch" directory
def generate_chunks() -> List[Dict[str, Any]]:
    all_chunk_data = []
    md_files = list(Path("scratch").glob("*.md"))
    print(f"Processing {len(md_files)} markdown files...")
    for i, md_path in enumerate(md_files, 1):
        md_texts = md_path.read_text()
        file_chunks = chunk(md_texts, source_file=md_path.name)
        all_chunk_data.extend(file_chunks)
        print(f"[{i}/{len(md_files)}] {md_path.name}: {len(file_chunks)} chunks")
    return all_chunk_data

""" Testing to see if the output actually comes out as expected """
# output_path = Path("scratch/chunks.txt")
# with output_path.open("w") as f:
#     for i, chunk in enumerate(chunks):
#         f.write(f"\n\n===== CHUNK {i} =====\n\n")
#         f.write(chunk)
#         f.write("\n")

def create_db_connection():
    # connect to the PostgreSQL database, casting the connection from variable -> to a vector type for psycopg1
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
        source_file VARCHAR(254),
        embedding vector(384) NOT NULL
      )
    ''')

    """ Index types for pgvector """
    # Approximate Nearest Neighbor (ANN) indexes for faster search MEDIUM to LARGE datasets, bad space
    cur.execute('CREATE INDEX IF NOT EXISTS embeddings ON items USING hnsw (embedding vector_l1_ops)')
    # Cluster-based ANN 
    # cur.execute('CREATE INDEX ON items USING ivfflat (embedding vector_l1_ops) WITH (lists = 100)')
    
    return conn, cur

# Extract chunk texts from the all_chunks dictionaries which contains texts + other metadata
def get_chunk_text(chunks):
    if chunks and isinstance(chunks[0], dict):
        chunk_texts = [chunk["chunk_text"] for chunk in chunks]
    else:
        chunk_texts = chunks
    return chunk_texts

def get_embeddings(chunk_texts):
    return model.encode(chunk_texts)

# Save embeddings to a file with chunk info.
def save_embeddings_file(chunks, embeddings, output_file="embeddings.txt"):
    with open(output_file, "w") as f:
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_info = f"Chunk {i}"
            if isinstance(chunk, dict):
                chunk_info = f"Chunk {i} (Source: {chunk.get('source_file', 'N/A')})"
            f.write(f"{chunk_info}:\n{emb.tolist() if hasattr(emb, 'tolist') else emb}\n\n")

def insert_chunks_and_embeddings_to_db(chunks, embeddings, cur, conn):
    # Prepare batch data
    batch_data = []
    for chunk, emb in zip(chunks, embeddings):
        if isinstance(chunk, dict):
            chunk_index = chunk["chunk_index"]
            chunk_text = chunk["chunk_text"]
            chunk_length = chunk["chunk_length"]
            source_file = chunk["source_file"]
        else:
            chunk_index = 0
            chunk_text = chunk
            chunk_length = len(chunk)
            source_file = None
        
        batch_data.append((chunk_index, chunk_text, chunk_length, source_file, emb.tolist() if hasattr(emb, "tolist") else emb))
    
    # Batch insert for better performance
    print(f"Inserting {len(batch_data)} chunks into database...")
    cur.executemany('''
        INSERT INTO items (chunk_index, chunk_text, chunk_length, source_file, embedding) 
        VALUES (%s, %s, %s, %s, %s::vector)
    ''', batch_data)
    conn.commit()
    print(f"Successfully inserted {len(batch_data)} chunks.")


def main():
    print("Starting chunking and embedding process...")
    conn, cur = create_db_connection()
    
    print("Generating chunks from markdown files...")
    chunks = generate_chunks()
    print(f"Generated {len(chunks)} total chunks.")
    
    chunk_texts = get_chunk_text(chunks)
    print(f"Generating embeddings for {len(chunk_texts)} chunks (this may take a while)...")
    embeddings = get_embeddings(chunk_texts)
    print("Embeddings generated successfully.")
    
    print("Saving embeddings to file...")
    save_embeddings_file(chunks, embeddings)

    insert_chunks_and_embeddings_to_db(chunks, embeddings, cur, conn)
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
