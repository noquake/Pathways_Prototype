# Local RAG Pipeline with Docling and PostgreSQL + pgvector

A lightweight **Retrieval-Augmented Generation** system using:

- **Docling** for PDF and Word document ingestion and generation of various files for later LLM ingestion (primarily .md at this stage 12/25)
- **SentenceTransformers** for embedding generation
- **Docker** spinning containers for embedding and (future) usage statistics as well as pgadmin for simple monitoring of database activity via docker-compose
- **PostgreSQL + PGVector** preferred database technology for this project and pjvector for injection of embeddings into containers

This project has been designed to be as clear as possible in its execution and debugging. It streams the generation of chunks, embeddings and their injection the docker database to reduce memory usage.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- PostgreSQL with pgvector extension

---

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install Python packages
pip install -r requirements.txt
```

### 2. Spin up psql database and administration containers

```bash
docker compose up -d
docker ps # check status of containers
```
