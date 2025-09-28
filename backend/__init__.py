try:
	# prefer the main agents module if present
	from .agents import create_agent_coordinator as root_agent
except Exception:
	# fallback to archived agents module (post-merge compatibility)
	from .archive.agents import create_agent_coordinator as root_agent

__all__ = ["root_agent", "AgentCoordinator"]
