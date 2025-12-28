from pathlib import Path
import psycopg2
from pgvector.psycopg2 import register_vector

from typing import Iterator, List, Dict, Any

from sentence_transformers import SentenceTransformer # type: ignore
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

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

chunk_total = 0

# Retrieve docling-ized md files, generate chunks and append them to an all_chunks list
def generate_chunks(md_dir: str, chunker: HybridChunker) -> Iterator:
    md_dir = Path("/data")
    md_files = (md_dir.glob("*.md"))

    assert md_files, f"No markdown files found in ({md_dir})"

    for file in md_files:
        doc = DoclingDocument(text=file.read_text(), source_file=str(file))

        for raw_chunk in chunker.chunk(dl_doc=doc):
            chunk_total += 1
            yield raw_chunk


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

def get_embedding(chunk):
    return model.encode(chunk)


def insert_chunk_and_embedding_to_db(chunk, embedding, cur, conn):
    # Prepare batch data
    batch_data = []
    for chunk, emb in zip(chunk, embedding):
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
    print(f"Successfully inserted {len(batch_data)} chunks.\n")


def main():
    print("Creating connection to PGVector database...\n")
    conn, cur = create_db_connection()
    
    print("Starting for loop to stream chunk generation -> contextualization -> embedding -> DB insertion...\n")
    
    for raw_chunk in generate_chunks("/data", chunker):
        print(f"Processing chunk #{chunk_total}...\n")
        chunk = chunker.contextualize(raw_chunk)
        print(f"Chunk #{chunk_total} contextualized. Now embedding...\n")
        emb = get_embedding(chunk)
        print(f"Chunk #{chunk_total} embedded. Now inserting into DB...\n")
        insert_chunk_and_embedding_to_db(chunk, emb, cur, conn)
        print(f"Chunk #{chunk_total} inserted into DB.\n")

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