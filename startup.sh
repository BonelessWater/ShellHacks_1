#!/bin/bash

# Azure App Service startup script for Node.js primary + FastAPI secondary
echo "🚀 Starting AgentZero - Node.js Primary Architecture"
echo "📍 Current directory: $(pwd)"
echo "📂 Directory contents:"
ls -la

# Set environment variables
export NODE_ENV=production
export PORT=3000
export REACT_APP_API_URL=http://localhost:8000

# Install Node.js dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install --production --silent
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Node.js dependencies"
        exit 1
    fi
else
    echo "✅ Node.js dependencies already installed"
fi

# Check if FastAPI directory exists
if [ -d "api" ]; then
    echo "🐍 Setting up FastAPI backend..."
    cd api
    
    # Install Python dependencies
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python dependencies"
        exit 1
    fi
    
    # Start FastAPI backend on port 8000 in background
    echo "🔄 Starting FastAPI backend on port 8000..."
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 &
    FASTAPI_PID=$!
    echo "🐍 FastAPI started with PID: $FASTAPI_PID"
    
    # Wait a moment for FastAPI to start
    sleep 3
    
    # Check if FastAPI is running
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ FastAPI backend is healthy"
    else
        echo "⚠️  FastAPI backend health check failed, but continuing..."
    fi
    
    cd ..
else
    echo "⚠️  No FastAPI backend found, running Node.js only"
fi

# Verify React build exists
if [ -d "build" ]; then
    echo "✅ React build directory found"
    echo "📊 Build contents:"
    ls -la build/ | head -5
else
    echo "❌ No React build directory found"
    exit 1
fi

# Check if server.js exists
if [ -f "server.js" ]; then
    echo "✅ Node.js server.js found"
else
    echo "❌ server.js not found"
    exit 1
fi

# Start Node.js server on port 3000 (primary service)
echo "🟢 Starting Node.js Express server on port 3000..."
echo "🌐 Environment: $NODE_ENV"
echo "🔗 API URL: $REACT_APP_API_URL"

# Trap signals to clean up background processes
trap 'echo "🛑 Stopping services..."; kill $FASTAPI_PID 2>/dev/null; exit' SIGTERM SIGINT

# Start the main Node.js server
node server.js

# If we reach here, the server stopped
echo "🔴 Node.js server stopped"