#!/bin/bash

echo "Starting ShellHack MCP Server..."

# Start MCP server in background
python -m shellhacks.mcp_server &
SERVER_PID=$!

echo "MCP Server started with PID: $SERVER_PID"

# Wait for server to be ready
sleep 3

# Run client examples
echo "Running MCP client tests..."
python -m shellhacks.mcp_client

# Keep server running
wait $SERVER_PID