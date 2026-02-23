from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import subprocess
import sys

from rag.scrape_docs import main as scrape_docs
from rag.transform_data import main as transform_data
from services.backend.rag.docling_chunk_supabase import main as docling_chunk_to_supabase

from rag_semantic.semantic_chunk_supabase import main as semantic_chunk_supabase
from rag_semantic.semantic_query import main as semantic_query

def spin_up_docker():
    # subprocess.run(["docker", "compose", "up", "-d", ])
    subprocess.run(["docker", "compose", "-f", "docker-compose.db.yml", "up", "-d"])

def main():
    from sentence_transformers import SentenceTransformer
    """Prepare data and ingest chunks into Supabase."""

    # create and start all docker containers ONLY UNCOMMENT WHEN TESTING LOCALLY
    # print("Spinning up docker containers...\n")
    # spin_up_docker()
    # print("Docker containers are up and running.\n")

    # scrape documents from specified sources
    # print("Starting document scraping...\n")
    # scrape_docs()
    # print("Document scraping complete.\n")
    
    # transform the retrieved documents into usable data formats
    # print("Transforming data for RAG...\n")
    # transform_data()
    # print("Data transformation complete.\n")

    # chunk and embed the transformed data for use in retrieval-augmented generation
    print("Starting semantic chunking and embedding...\n")
    semantic_chunk_supabase()
    print("Semantic chunking and embedding complete.\n")



if __name__ == "__main__": 
    main()
