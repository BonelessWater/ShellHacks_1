#!/bin/bash

# 🚀 SETUP FOR LOCALHOST:3000

echo "🛑 Stopping existing containers..."
docker-compose down

echo "📦 Installing http-proxy-middleware in frontend..."
cd frontend
npm install http-proxy-middleware
cd ..

echo "🏗️  Building for development (React dev server on :3000)..."
docker-compose -f docker-compose.dev.yml build --no-cache

echo "🚀 Starting development environment..."
docker-compose -f docker-compose.dev.yml up -d

echo "📋 Container status:"
docker-compose -f docker-compose.dev.yml ps

echo "🔗 URLs:"
echo "   Frontend (React Dev): http://localhost:3000"
echo "   Backend API:          http://localhost:8000"
echo "   Health Check:         http://localhost:8000/health"

echo "📋 View logs:"
echo "   docker-compose -f docker-compose.dev.yml logs -f"

# Alternative: Production with Express on port 3000
echo ""
echo "🏭 For production (Express server on :3000):"
echo "   docker-compose build --no-cache"
echo "   docker-compose up -d"