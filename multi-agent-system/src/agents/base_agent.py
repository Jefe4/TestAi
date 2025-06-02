# src/agents/base_agent.py
"""Defines the abstract base class for all specialized agents in the multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any # Added Dict, Optional, Any for type hinting
# Assuming multi-agent-system is the root of the PYTHONPATH or src is added to path
# For a direct relative import if this file is run as part of a package:
from ..utils.logger import get_logger

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized agents.
    Each agent in the system should inherit from this class and implement
    the required abstract methods.
    """
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_name: The name of the agent (e.g., "deepseek_coder", "gemini_pro").
            config: An optional dictionary containing agent-specific configuration
                    (e.g., API key, model name, temperature settings).
        """
        self.agent_name = agent_name
        self.config = config if config is not None else {}
        # Use the project's custom logger
        self.logger = get_logger(f"BaseAgent.{self.agent_name}")

        # More careful logging of config, avoid logging potentially sensitive full config
        log_config_summary = {k: v for k, v in self.config.items() if k not in ['api_key', 'api_token']}
        self.logger.info(f"Agent '{self.agent_name}' initialized. Config summary: {log_config_summary}")


    @abstractmethod
    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a given query data using the agent's capabilities and returns a response.
        This method must be implemented by all specialized agent subclasses.

        Args:
            query_data: A dictionary containing the query and any other relevant data.
                        Example: {"query": "Generate python code for a factorial function", "details": {...}}

        Returns:
            A dictionary containing the agent's response.
            Example: {"status": "success", "result": "def factorial(n): ..."}
        """
        raise NotImplementedError(f"{self.__class__.__name__}.process_query is not implemented.")

    def get_name(self) -> str:
        """
        Returns the name of the agent.

        Returns:
            The agent's name.
        """
        return self.agent_name

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the agent's capabilities.
        This method can be overridden by subclasses to declare specific capabilities.

        Returns:
            A dictionary describing the agent's capabilities.
        """
        return {"description": "Base agent - capabilities not defined"}

    def format_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats the raw response data from the agent.
        This can be overridden by subclasses to provide specific formatting.

        Args:
            response_data: The raw data returned by the agent's processing logic.

        Returns:
            The formatted response data.
        """
        return response_data
