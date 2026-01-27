FROM python:3.11-slim

WORKDIR /app

# Install OS deps
# Install OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpq-dev \
    curl \
    netcat-openbsd \
    # NEEDED for Docling / OpenCV:
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entrypoint script and make it executable
# COPY entrypoint.sh /entrypoint.sh
# RUN chmod +x /entrypoint.sh


COPY . .

# Set the entrypoint
# ENTRYPOINT ["/entrypoint.sh"]
# CMD ["python", "docling_chunk.py"]
CMD ["python", "app.py"]