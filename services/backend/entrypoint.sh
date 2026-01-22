#!/bin/bash
set -e

# OPTIONAL: You can control this via docker-compose env vars
if [ "$RUN_INGESTION" = "true" ]; then
    echo "--- WORKER MODE: Starting Data Ingestion ---"
    python rag/docling_chunk.py
    echo "--- Ingestion Complete ---"
else
    echo "--- SERVER MODE: Skipping Ingestion ---"
fi

echo "--- Starting Web Server ---"
# This replaces the shell process with Uvicorn, keeping signals working
exec uvicorn main:app --host 0.0.0.0 --port 8000