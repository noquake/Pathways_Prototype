from pathlib import Path
from sentence_transformers import SentenceTransformer
import psycopg2

model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_markdown(md_texts, max_len=1000):
    chunks = []
    current = []

    for line in md_texts.split("\n"):
        if line.startswith("# "):
            if current:
                chunks.append("\n".join(current))
                current = []
        current.append(line)
        if sum(len(l) for l in current) + len(line) > max_len:
            chunks.append("\n".join(current))
            current = []
    if current:
        chunks.append("\n".join(current))
    return chunks

for md_path in Path("scratch").glob("*.md"):
    md_texts = md_path.read_text()
    chunks = chunk_markdown(md_texts)

    print(md_path.name, len(chunks))


""" Testing to see if the output actually comes out as expected """
output_path = Path("scratch/chunks.txt")
with output_path.open("w") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"\n\n===== CHUNK {i} =====\n\n")
        f.write(chunk)
        f.write("\n")

def embed_chunks(chunks):
    embeddings = model.encode(chunks)
    with open("embeddings.txt", "w") as f:
        for i, emb in enumerate(embeddings):
            f.write(f"Chunk {i}:\n{emb.tolist() if hasattr(emb, 'tolist') else emb}\n\n")
    
embed_chunks(chunks)

# conn = psycopg2.connect("dbname=pathways user=admin password=password host=your_host")