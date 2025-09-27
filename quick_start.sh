#!/bin/bash
# quick-start.sh - Start your ShellHacks Invoice Frontend with Docker

set -e

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting ShellHacks Invoice Frontend${NC}"
echo "=============================================="

# Check if Docker is running
echo -e "${YELLOW}🔍 Checking Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}❌ docker-compose.yml not found. Please run this script from the project root.${NC}"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo -e "${RED}❌ frontend directory not found. Please make sure you're in the project root.${NC}"
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}📝 Creating .env file...${NC}"
    cat > .env << 'EOF'
# Frontend Environment Variables
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WEBSOCKET_URL=ws://localhost:8000
REACT_APP_ENVIRONMENT=development

# Optional: Backend API Key (uncomment if needed)
# GOOGLE_API_KEY_0=your_google_api_key_here
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
docker-compose down > /dev/null 2>&1 || true

# Build and start the frontend
echo -e "${YELLOW}🔨 Building and starting frontend...${NC}"
echo "This may take a few minutes on first run..."

# Build with progress
docker-compose build frontend

# Start the service
echo -e "${YELLOW}🚀 Starting the frontend service...${NC}"
docker-compose up -d frontend

# Wait for the service to be ready
echo -e "${YELLOW}⏳ Waiting for frontend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

# Check if service is running
if docker-compose ps frontend | grep -q "Up"; then
    echo ""
    echo "=============================================="
    echo -e "${GREEN}🎉 SUCCESS! Frontend is running${NC}"
    echo ""
    echo -e "${BLUE}📱 Frontend URL:${NC} http://localhost:3000"
    echo -e "${BLUE}🐳 Docker Status:${NC} Running"
    echo ""
    echo -e "${YELLOW}📋 Useful Commands:${NC}"
    echo "  View logs:    docker-compose logs -f frontend"
    echo "  Stop:         docker-compose down"
    echo "  Restart:      docker-compose restart frontend"
    echo ""
    echo -e "${GREEN}✨ Your ShellHacks Invoice app is ready!${NC}"
    echo "=============================================="
    
    # Ask if user wants to see logs
    echo ""
    read -p "Would you like to view the logs? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}📊 Showing logs (Press Ctrl+C to exit):${NC}"
        docker-compose logs -f frontend
    fi
else
    echo -e "${RED}❌ Frontend failed to start. Check the logs:${NC}"
    docker-compose logs frontend
    exit 1
fi