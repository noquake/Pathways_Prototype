from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

model: Optional[SentenceTransformer] = None

def get_embeddings(chunk_texts: List[str]):
    """Return embeddings for a list of texts."""
    global model
    if model is None:
        # Prefer offline/local cache to keep service startup independent of network.
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        # model = SentenceTransformer("all-mpnet-base-v2")
        # model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        # model = SentenceTransformer("ncbi/MedCPT-Query-Encoder")
    return model.encode(chunk_texts)