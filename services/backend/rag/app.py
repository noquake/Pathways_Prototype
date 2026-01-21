from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import subprocess
import sys

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse



from rag.transform_data import main as transform_data
from rag.default_chunk import main as default_chunk
from rag.docling_chunk import main as docling_chunk

def spin_up_docker():
    # subprocess.run(["docker", "compose", "up", "-d", ])
    subprocess.run(["docker", "compose", "-f", "docker-compose.test.yml", "up", "-d"])

def main():
    """ spin up all docker containers ->  prepare data for RAG -> chunk and embed data into pathways_db """
    # create and start all docker containers
    print("Spinning up docker containers...\n")
    spin_up_docker()
    print("Docker containers are up and running.\n")
    
    # transform the retrieved documents into usable data formats
    print("Transforming data for RAG...\n")
    transform_data()
    print("Data transformation complete.\n")

    # chunk and embed the transformed data for use in retrieval-augmented generation
    print("Starting document chunking and embedding...\n")
    docling_chunk()
    # default_chunk()
    print("Document chunking and embedding complete.\n")



if __name__ == "__main__": 
    main()
