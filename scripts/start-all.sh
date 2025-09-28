#!/usr/bin/env bash
# scripts/start-all.sh - Start backend (uvicorn) and frontend (react-scripts) for local development

set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/frontend"

echo "Starting backend (uvicorn) on port 8000..."
poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: ${BACKEND_PID} (logs -> backend.log)"

trap 'echo "Stopping backend (${BACKEND_PID})"; kill ${BACKEND_PID} || true; exit' INT TERM EXIT

echo "Waiting for backend to start..."
sleep 5

echo "Starting frontend (npm start) in ${FRONTEND_DIR}..."
cd "$FRONTEND_DIR"
npm start

# When frontend exits, trap will run and kill backend
