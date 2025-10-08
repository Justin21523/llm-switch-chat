#!/bin/bash

# Development startup script

set -e

echo "Starting LLM Switch Chat in development mode..."

# Activate conda environment if specified
if [ ! -z "$CONDA_ENV" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
fi

# Set default environment variables
export MODEL_BACKEND=${MODEL_BACKEND:-hf}
export MODEL_ID=${MODEL_ID:-Qwen/Qwen2-7B-Instruct}
export ENABLE_RAG=${ENABLE_RAG:-false}
export DEBUG=${DEBUG:-true}

echo "Configuration:"
echo "  Backend: $MODEL_BACKEND"
echo "  Model: $MODEL_ID"
echo "  RAG: $ENABLE_RAG"
echo "  Debug: $DEBUG"

# Install package in editable mode
echo "Installing package in editable mode..."
pip install -e .

# Start API server in background
echo "Starting API server..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 5

# Start Gradio UI
echo "Starting Gradio UI..."
python -m app.ui.gradio_app --api-url http://localhost:8000 --port 7860 &
UI_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Shutting down..."
    kill $API_PID $UI_PID 2>/dev/null || true
    wait
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Wait for processes
echo "Services started:"
echo "  API: http://localhost:8000"
echo "  UI: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop all services"

wait
```# LLM Switch Chat - 完整實作代碼

## 1. 後端核心檔案

### backend/app/__init__.py
```python
"""LLM Switch Chat Application Package."""