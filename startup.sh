#!/bin/bash
set -euo pipefail

echo "🚀 Starting Node + FastAPI app..."

# --- Step 1: Root npm install ---
echo "📦 Installing root Node.js dependencies..."
npm ci || npm install --no-audit --no-fund

# --- Step 2: Frontend build ---
echo "📦 Installing frontend dependencies and building React app..."
cd frontend
npm ci || npm install --no-audit --no-fund
npm run build
cd ..

# --- Step 3: Python dependencies ---
if [ -f "requirements.txt" ]; then
  echo "🐍 Installing Python dependencies..."
  pip install --no-cache-dir -r requirements.txt
fi

# --- Step 4: Start FastAPI backend ---
echo "🐍 Starting FastAPI backend (backend/main.py) on :8000..."
( uvicorn backend.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips="*" ) &

# --- Step 5: Start Node server ---
echo "🟩 Starting Node/Express server.js on ${PORT:-8080}..."
exec node server.js
