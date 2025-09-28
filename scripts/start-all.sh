#!/usr/bin/env bash
# scripts/start-all.sh - Start backend (uvicorn) and frontend (react-scripts) for local development

set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/frontend"

echo "Starting backend (uvicorn) on port 8000..."
# Prefer Poetry environment if available so uvicorn is found consistently
if command -v poetry >/dev/null 2>&1; then
	echo "Found poetry, starting backend with 'poetry run uvicorn'"
	poetry run uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
else
	echo "Poetry not found, falling back to 'python -m uvicorn' (may require uvicorn installed in this Python)"
	python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
fi
BACKEND_PID=$!
echo "Backend PID: ${BACKEND_PID} (logs -> backend.log)"

trap 'echo "Stopping backend (${BACKEND_PID})"; kill ${BACKEND_PID} || true; exit' INT TERM EXIT

echo "Waiting a moment for backend to become ready..."
sleep 1

echo "Starting frontend (npm start) in ${FRONTEND_DIR}..."
cd "$FRONTEND_DIR"
npm start

# When frontend exits, trap will run and kill backend
