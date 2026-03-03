from typing import Dict, Any

EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "minilm": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "table": "pathway_chunks_linilm",
        "rpc_function": "match_chunks_minilm",
        "description": "Fast, lightweight (baseline)"
    },
    "pubmedbert": {
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "dimension": 768,
        "table": "pathway_chunks_pubmedbert",
        "rpc_function": "match_chunks_pubmedbert",
        "description": "Medical domain-specific"
    },
    "mpnet": {
        "model_name": "all-mpnet-base-v2",
        "dimension": 768,
        "table": "pathway_chunks_mpnet",
        "rpc_function": "match_chunks_mpnet",
        "description": "General purpose, better than MiniLM"
    },
    "medcpt": {
        "model_name": "ncbi/MedCPT-Article-Encoder",
        "query_model_name": "ncbi/MedCPT-Query-Encoder",  # ← dual encoder
        "dimension": 1024,
        "table": "pathway_chunks_medcpt",
        "rpc_function": "match_chunks_medcpt",
        "description": "Clinical Q&A specialized"
    }
}

DEFAULT_MODEL = "minilm"