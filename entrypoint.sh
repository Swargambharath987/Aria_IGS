#!/bin/bash
set -e

OLLAMA_HOST="http://ollama:11434"

echo "==> Waiting for Ollama to be ready..."
until curl -s "$OLLAMA_HOST" > /dev/null 2>&1; do
    sleep 2
done
echo "==> Ollama is ready."

echo "==> Pulling models if not already present..."
curl -s "$OLLAMA_HOST/api/pull" -d '{"name":"qwen2.5:7b"}' > /dev/null
curl -s "$OLLAMA_HOST/api/pull" -d '{"name":"nomic-embed-text"}' > /dev/null
echo "==> Models ready."

# Run ingestion only if ChromaDB is empty
CHROMA_DIR="data/chroma_db"
if [ ! "$(ls -A $CHROMA_DIR 2>/dev/null)" ]; then
    echo "==> ChromaDB is empty. Running ingestion (this takes a few minutes)..."
    OLLAMA_HOST_URL="$OLLAMA_HOST" python ingest/ingest.py
    echo "==> Ingestion complete."
else
    echo "==> Knowledge base already exists. Skipping ingestion."
fi

echo "==> Starting Streamlit..."
streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
