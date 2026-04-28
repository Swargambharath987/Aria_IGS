FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ingest/ ./ingest/
COPY rag/ ./rag/
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/raw/ ./data/raw/

# Create chroma_db directory (will be populated at runtime)
RUN mkdir -p data/chroma_db

# Copy entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8501

ENTRYPOINT ["./entrypoint.sh"]
