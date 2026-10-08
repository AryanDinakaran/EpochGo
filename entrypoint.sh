#!/bin/bash
set -e

echo "=================================================="
echo "⚡ Starting EpochGo — Autonomous Multi-Agent AutoML"
echo "=================================================="

OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MODEL_NAME="${OLLAMA_MODEL:-qwen2.5-coder:7b}"

echo "🔗 Checking connection to Ollama at: $OLLAMA_URL"

# Wait up to 30 seconds for Ollama server to become responsive if starting together
MAX_RETRIES=15
COUNTER=0
while [ $COUNTER -lt $MAX_RETRIES ]; do
    if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        echo "✅ Connected to Ollama server!"
        break
    fi
    echo "⏳ Waiting for Ollama ($COUNTER/$MAX_RETRIES)..."
    sleep 2
    COUNTER=$((COUNTER + 1))
done

# Check if model exists, if not pull in background or foreground
if curl -s "${OLLAMA_URL}/api/tags" | grep -q "$MODEL_NAME"; then
    echo "✅ Model '$MODEL_NAME' is ready."
else
    echo "📥 Model '$MODEL_NAME' not found in Ollama."
    echo "⚡ Initiating background pull for '$MODEL_NAME'..."
    curl -s -X POST "${OLLAMA_URL}/api/pull" -d "{\"name\": \"$MODEL_NAME\"}" > /dev/null 2>&1 &
fi

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "🚀 Starting EpochGo Web Server on http://0.0.0.0:8000"
echo "🌐 Open your browser at: http://localhost:8000"
echo "=================================================="

exec uvicorn epoch_go.main:app --host 0.0.0.0 --port 8000
