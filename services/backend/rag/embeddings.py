from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from .models import EMBEDDING_MODELS

_model_cache = {}

def get_embeddings(texts: List[str], model_key: str = "mpnet"):
    global _model_cache
    
    if model_key not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Choose from: {list(EMBEDDING_MODELS.keys())}")
    
    config = EMBEDDING_MODELS[model_key]
    model_name = config["model_name"]
    
    # Cache models to avoid reloading
    if model_key not in _model_cache:
        print(f"Loading {model_name}...")
        _model_cache[model_key] = SentenceTransformer(model_name)
    
    model = _model_cache[model_key]
    return model.encode(texts)