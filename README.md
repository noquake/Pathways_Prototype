# Local RAG Pipeline with PostgreSQL + pgvector

A lightweight **Retrieval-Augmented Generation (RAG)** system using:

- **Docling** for document ingestion and parsing
- **SentenceTransformers** for embeddings
- **PostgreSQL + pgvector** for vector storage and similarity search

Project is fully local, cross-platform (Linux/macOS), and optimized for small footprint deployment.

---

## Features

- Parse PDFs, DOCX, TXT, and other documents with Docling
- Chunk text automatically for semantic search
- Generate dense embeddings using SentenceTransformers
- Store text and embeddings in PostgreSQL using pgvector
- Perform vector similarity search to retrieve relevant context
- Optionally serve queries via a REST API using FastAPI + Uvicorn

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional but recommended)
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
