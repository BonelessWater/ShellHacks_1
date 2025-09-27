from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from mcp import Client
import asyncio

class ShellHackAgent:
    def __init__(self):
        # Initialize Google ADK agent
        self.llm_agent = LlmAgent(
            name="shellhack_assistant",
            model="gemini-2.0-flash-exp",
            instruction="You are a helpful hackathon assistant with MCP capabilities.",
            description="ShellHack competition assistant with MCP integration",
            tools=[google_search]
        )
        
        # Initialize MCP client
        self.mcp_client = Client()
        
    async def setup_mcp(self):
        """Setup MCP connection"""
        await self.mcp_client.connect("http://localhost:8000")
        
    async def process_request(self, user_input: str):
        """Process user request using both LLM and MCP"""
        # Get LLM response
        llm_response = await self.llm_agent.generate(user_input)
        
        # Enhance with MCP tools if needed
        if "search" in user_input.lower():
            mcp_result = await self.mcp_client.call_tool(
                "search_documentation",
                query=user_input
            )
            return {
                "llm_response": llm_response,
                "mcp_enhancement": mcp_result
            }
        
        return {"llm_response": llm_response}

# Create root agent instance
root_agent = ShellHackAgent()