from mcp import Client
import asyncio
from typing import Optional

class ShellHackMCPClient:
    def __init__(self):
        self.client = None
        
    async def connect(self, server_url: str = "http://localhost:8000"):
        """Connect to MCP server"""
        self.client = Client()
        await self.client.connect(server_url)
        print(f"Connected to MCP server at {server_url}")
        
    async def call_tool(self, tool_name: str, **kwargs):
        """Call a tool on the MCP server"""
        if not self.client:
            raise RuntimeError("Client not connected")
        
        result = await self.client.call_tool(tool_name, **kwargs)
        return result
        
    async def get_completion(self, prompt: str, context: Optional[dict] = None):
        """Get AI completion with context"""
        if not self.client:
            raise RuntimeError("Client not connected")
            
        response = await self.client.complete(
            prompt=prompt,
            context=context or {}
        )
        return response

# Example usage
async def main():
    client = ShellHackMCPClient()
    await client.connect()
    
    # Call a tool
    result = await client.call_tool(
        "search_documentation",
        query="API endpoints"
    )
    print(result)
    
    # Get completion
    response = await client.get_completion(
        "How do I implement authentication?",
        context={"project": "shellhack"}
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())