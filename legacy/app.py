from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import sys

from services.backend.rag.oldchunking import main as chunking


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
    print("Starting chunking and embedding...\n")
    chunking()
    print("Chunking and embedding complete.\n")



if __name__ == "__main__": 
    main()
