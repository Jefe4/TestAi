# src/agents/deepseek_agent.py
"""Specialized agent for interacting with DeepSeek AI models."""

from typing import Dict, Any, Optional

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager
    # Logger is typically available via self.logger from BaseAgent
except ImportError:
    # This block is for fallback if the script is run in a way that top-level packages aren't recognized
    # For instance, if you run `python deepseek_agent.py` directly from the agents directory.
    import sys
    import os
    # Add project root to sys.path to allow for absolute imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore


class DeepSeekAgent(BaseAgent):
    """
    An agent that utilizes DeepSeek AI models for various tasks like
    code generation, text completion, and complex reasoning.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DeepSeekAgent.

        Args:
            agent_name: The name of the agent.
            api_manager: An instance of APIManager to handle API calls.
            config: Optional configuration dictionary for the agent.
                    Expected keys: "model", "max_tokens", "temperature", "default_system_prompt".
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager

        # Default model if not specified in config
        self.model = self.config.get("model", "deepseek-coder")
        self.logger.info(f"DeepSeekAgent '{self.agent_name}' initialized with model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the DeepSeekAgent.
        """
        return {
            "description": "Agent for deep reasoning, complex problem solving, and code analysis using DeepSeek.",
            "capabilities": ["text_generation", "code_generation", "code_analysis", "complex_reasoning"],
            "models_supported": ["deepseek-coder", "deepseek-llm"] # Example, can be dynamic
        }

    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the DeepSeek API.

        Args:
            query_data: A dictionary containing the query details.
                        Expected keys:
                        - "prompt" (str): The user's actual query.
                        - "system_prompt" (Optional[str]): Custom system prompt for this query.
                                                           Overrides default if provided.
                        - Other potential keys for future use: "conversation_history", "temperature_override", etc.

        Returns:
            A dictionary containing the status of the operation and the response content or error message.
        """
        user_prompt = query_data.get("prompt")
        if not user_prompt:
            self.logger.error("User prompt is missing in query_data.")
            return {"status": "error", "message": "User prompt missing"}

        # Determine system prompt: use query-specific, then config default, then a generic default.
        default_system_prompt_from_config = self.config.get("default_system_prompt", "You are a helpful AI assistant specialized in coding and complex problem-solving.")
        system_prompt = query_data.get("system_prompt", default_system_prompt_from_config)

        self.logger.info(f"Processing query for DeepSeekAgent '{self.agent_name}' with prompt: '{user_prompt[:100]}...'")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 4096)), # Allow query-time override
            "temperature": query_data.get("temperature", self.config.get("temperature", 0.3)), # Allow query-time override
            # Other parameters like 'stream', 'top_p' can be added here from config or query_data
        }

        self.logger.debug(f"DeepSeek API payload: {payload}")

        # Make the API call via APIManager
        # Assuming 'deepseek' is a configured service in APIManager
        # and 'chat/completions' is the correct endpoint.
        response_data = self.api_manager.make_request(
            service_name='deepseek',
            endpoint='chat/completions',
            method="POST",
            data=payload
        )

        # Handle the response from APIManager
        if response_data.get("error"):
            self.logger.error(f"API request failed for DeepSeek: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data) # Include more details if available
            }

        # Extract the relevant text from DeepSeek's response
        try:
            # Typical DeepSeek API response structure:
            # { "choices": [ { "index": 0, "message": { "role": "assistant", "content": "response text" }, "finish_reason": "stop" } ] ... }
            extracted_text = response_data.get("choices", [{}])[0].get("message", {}).get("content")
            finish_reason = response_data.get("choices", [{}])[0].get("finish_reason")

            if extracted_text is None:
                self.logger.error(f"Failed to extract content from DeepSeek response. Response: {response_data}")
                return {"status": "error", "message": "Invalid response structure from DeepSeek API."}

            self.logger.info(f"Successfully received and parsed response from DeepSeek for prompt: '{user_prompt[:100]}...'")
            return {
                "status": "success",
                "content": extracted_text,
                "finish_reason": finish_reason,
                "usage": response_data.get("usage") # Include token usage if available
            }
        except (IndexError, KeyError, TypeError) as e:
            self.logger.error(f"Error parsing DeepSeek response: {e}. Response data: {response_data}")
            return {"status": "error", "message": f"Error parsing DeepSeek response: {e}"}


if __name__ == '__main__':
    # This block is for basic demonstration and testing.
    # It requires APIManager and BaseAgent to be correctly set up.

    # Setup a dummy APIManager and logger for local testing
    # In a real application, these would be part of the main system.
    from src.utils.logger import get_logger as setup_logger # type: ignore

    # Dummy APIManager that simulates responses
    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_DeepSeekTest")
            # Simulate service configs if APIManager's load_service_configs is complex for dummy
            self.service_configs = {
                "deepseek": {
                    "api_key": "dummy_deepseek_key",
                    "base_url": "https://api.deepseek.com/v1"
                }
            }


        def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            if service_name == "deepseek" and endpoint == "chat/completions":
                # Simulate a successful DeepSeek API response
                if "error" in data.get("messages")[1].get("content","").lower(): # Simulate error
                     return {"error": "Simulated API Error", "message": "The prompt contained 'error'", "status_code": 400}

                return {
                    "id": "chatcmpl_xxxxxxxxxxxxxxxxx",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": data.get("model", "deepseek-coder"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"This is a simulated response to: '{data['messages'][1]['content']}' using model {data.get('model')}."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(data['messages'][0]['content']) + len(data['messages'][1]['content']),
                        "completion_tokens": 50, # dummy value
                        "total_tokens": len(data['messages'][0]['content']) + len(data['messages'][1]['content']) + 50
                    }
                }
            return {"error": "Unknown service or endpoint in DummyAPIManager", "status_code": 404}

    print("--- Testing DeepSeekAgent ---")

    # Initialize dummy components
    dummy_api_manager = DummyAPIManager()
    agent_config = {
        "model": "deepseek-coder-test", # Test if config model is picked up
        "max_tokens": 100,
        "temperature": 0.5,
        "default_system_prompt": "You are a specialized DeepSeek test assistant."
    }

    deepseek_agent = DeepSeekAgent(
        agent_name="TestDeepSeekAgent001",
        api_manager=dummy_api_manager, # type: ignore
        config=agent_config
    )

    print(f"Agent Name: {deepseek_agent.get_name()}")
    print(f"Agent Capabilities: {deepseek_agent.get_capabilities()}")
    print(f"Agent Configured Model: {deepseek_agent.model}")

    # Test case 1: Simple query
    print("\n--- Test Case 1: Simple Query ---")
    query1_data = {"prompt": "What is the capital of France?"}
    response1 = deepseek_agent.process_query(query1_data)
    print(f"Response 1: {response1}")
    assert response1["status"] == "success"
    assert "Paris" in response1.get("content", "") or "simulated response" in response1.get("content", "")

    # Test case 2: Query with custom system prompt and parameters
    print("\n--- Test Case 2: Custom System Prompt & Params ---")
    query2_data = {
        "prompt": "Generate a short Python function to add two numbers.",
        "system_prompt": "You are a Python code generation expert.",
        "temperature": 0.7, # Test query-time override
        "max_tokens": 50 # Test query-time override
    }
    response2 = deepseek_agent.process_query(query2_data)
    print(f"Response 2: {response2}")
    assert response2["status"] == "success"
    assert "def add_numbers" in response2.get("content", "") or "simulated response" in response2.get("content", "")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = deepseek_agent.process_query(query3_data)
    print(f"Response 3: {response3}")
    assert response3["status"] == "error"
    assert response3["message"] == "User prompt missing"

    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = deepseek_agent.process_query(query4_data)
    print(f"Response 4: {response4}")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- DeepSeekAgent testing completed. ---")
    print("Note: The fallback import mechanism for BaseAgent/APIManager is primarily for isolated testing of this script.")
    print("In the full system, these should be resolved by Python's package structure.")
