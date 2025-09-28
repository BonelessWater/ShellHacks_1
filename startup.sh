#!/bin/bash
set -e

echo "🚀 Starting AgentZero deployment..."

# Set environment variables
export NODE_ENV=production
export PORT="${PORT:-8080}"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install --production

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
elif [ -f "api/requirements.txt" ]; then
    python -m pip install --upgrade pip
    python -m pip install -r api/requirements.txt
fi

# Start Python backend in background
echo "🐍 Starting Python FastAPI backend on port 8000..."
if [ -f "api/main.py" ]; then
    cd api
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
    cd ..
elif [ -f "backend/main.py" ]; then
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 1 &
elif [ -f "main.py" ]; then
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
fi

# Wait a moment for backend to start
sleep 5

# Start Node.js server (main process)
echo "🟢 Starting Node.js server on port $PORT..."
exec node server.js