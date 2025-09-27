#!/usr/bin/env python3
"""
Main entry point for the ShellHacks test assistant.
"""

import asyncio
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from shellhacks.agent import root_agent

# Configuration
APP_NAME = "shellhacks"
USER_ID = "test_user"
SESSION_ID = "test_session"


async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")
    
    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    final_response_text = "Agent did not produce a final response."
    
    # Execute the agent and find the final response
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
                break
    
    print(f"<<< Agent Response: {final_response_text}")


async def interactive_chat():
    """Run an interactive chat session with the test assistant."""
    print("Starting the test assistant...")
    print(f"Agent Name: {root_agent.name}")
    print(f"Model: {root_agent.model}")
    print(f"Description: {root_agent.description}")
    print("\nType 'quit' to exit the chat.\n")
    
    # Set up session and runner
    session_service = InMemorySessionService()
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    
    # Interactive chat loop
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input:
                await call_agent_async(
                    user_input,
                    runner=runner,
                    user_id=USER_ID,
                    session_id=SESSION_ID
                )
    except KeyboardInterrupt:
        print("\nShutting down the test assistant...")
    except Exception as e:
        print(f"Error running the agent: {e}")


def start():
    """
    Start the test assistant agent.
    """
    asyncio.run(interactive_chat())


if __name__ == "__main__":
    start()