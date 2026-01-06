FROM python:3.11-slim

WORKDIR /app

# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt