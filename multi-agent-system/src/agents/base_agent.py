# src/agents/base_agent.py
"""
Defines the abstract base class (ABC) for all specialized agents
within the multi-agent system. It establishes a common interface
that all concrete agent implementations must adhere to.
"""

from abc import ABC, abstractmethod # For creating abstract base classes and methods
from typing import Dict, Optional, Any # Standard typing modules

# Attempt to import the project's logger utility.
# Includes a fallback if run in an environment where relative imports fail.
try:
    from ..utils.logger import get_logger
except ImportError:
    import logging # Standard Python logging module for fallback
    def get_logger(name: str) -> logging.Logger: # type: ignore
        """Fallback basic logger if the project's get_logger is unavailable."""
        logger = logging.getLogger(name)
        if not logger.handlers: # Add a handler if none exists, to ensure messages are visible
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO) # Default log level
        return logger

class BaseAgent(ABC):
    """
    Abstract Base Class for all specialized agents.

    This class defines the fundamental structure and methods that any agent
    in the system must implement. It provides a common interface for the
    Coordinator to interact with different types of agents.

    Attributes:
        agent_name (str): The unique name of the agent instance.
        config (Dict[str, Any]): Agent-specific configuration parameters.
        logger (logging.Logger): A logger instance for this agent.
    """
    def __init__(self, agent_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_name: The name of the agent (e.g., "ClaudeAgent", "DeepSeekCoder").
                        This name is used for logging and identification.
            config: An optional dictionary containing agent-specific configurations,
                    such as API keys (though preferably managed by APIManager),
                    model names, default parameters (e.g., temperature), etc.
        """
        self.agent_name = agent_name
        self.config: Dict[str, Any] = config if config is not None else {}
        # Initialize a logger specific to this agent instance for better traceability.
        self.logger = get_logger(f"Agent.{self.agent_name}")
        
        # Log a summary of the configuration, excluding sensitive keys like 'api_key'.
        # This helps in debugging initialization without exposing secrets.
        log_config_summary = {k: v for k, v in self.config.items() if k.lower() not in ['api_key', 'apikey', 'api_token']}
        self.logger.info(f"Agent '{self.agent_name}' initialized. Config summary (sensitive keys excluded): {log_config_summary}")


    @abstractmethod
    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
        """
        Processes a given query using the agent's specific capabilities and returns a response.

        This is an abstract method and *must* be implemented by all concrete agent subclasses.
        It defines the core logic of how an agent handles a request.

        Args:
            query_data: A dictionary containing the query and any other relevant data
                        needed by the agent to process the request. The structure of
                        this dictionary can vary depending on the agent's needs.
                        Example: `{"prompt": "Generate Python code for a factorial function"}`
                                 `{"prompt_parts": ["Describe this image:", <image_data>]}` (for multimodal)

        Returns:
            A dictionary containing the agent's response. Typically, this includes
            a "status" field ("success" or "error") and other fields like "content"
            or "message".
            Example success: `{"status": "success", "content": "def factorial(n): ..."}`
            Example error: `{"status": "error", "message": "API call failed"}`
        """
        # This line ensures that subclasses must implement this method.
        raise NotImplementedError(f"{self.__class__.__name__}.process_query is an abstract method and must be implemented.")

    def get_name(self) -> str:
        """
        Returns the name of the agent instance.

        Returns:
            The agent's name as a string.
        """
        return self.agent_name

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the agent's capabilities.

        This method should be overridden by specialized agent subclasses to declare
        their specific functionalities, supported tasks, or models they can interact with.
        The structure of the returned dictionary can be flexible but might include
        keys like "description", "supported_tasks", "models", "input_formats", etc.

        Returns:
            A dictionary describing the agent's capabilities. The base implementation
            returns a generic message indicating capabilities are not defined.
        """
        self.logger.debug(f"Base get_capabilities called for {self.agent_name}. Subclass should override if specific capabilities are needed.")
        return {
            "description": f"Base agent '{self.agent_name}' - specific capabilities not defined. Override in subclass.",
            "capabilities": [] # Default to empty list
        }

    def format_response(self, response_data: Any) -> Dict[str, Any]: # Changed response_data type to Any
        """
        Standardizes or formats the raw response data from an agent's core processing logic.

        This method can be overridden by subclasses if they need to transform
        the raw output from an API or internal processing into a consistent
        response structure expected by the Coordinator or other components.
        The base implementation simply returns the data as is, assuming it's already
        in the desired dictionary format with a "status" key.

        Args:
            response_data: The raw data returned by the agent's processing logic.
                           This could be a string, dictionary, or other type.

        Returns:
            A dictionary representing the formatted response. It's recommended to
            include at least a "status" field ("success" or "error").
            If `response_data` is not already a dict, it should be wrapped in one.
        """
        if isinstance(response_data, dict) and "status" in response_data:
            return response_data
        # If response_data is not a dict or lacks status, wrap it.
        # This is a basic way to ensure a consistent return type.
        # Subclasses should ideally handle their specific raw response types more robustly.
        self.logger.warning(f"Response data for {self.agent_name} was not a dict with 'status'. Wrapping it. Original type: {type(response_data)}")
        return {"status": "unknown", "content": response_data}
