# src/agents/deepseek_agent.py
"""
Specialized agent for interacting with DeepSeek AI models, particularly
those focused on code generation and complex reasoning tasks.
"""

import asyncio
from typing import Dict, Any, Optional, List # Added List for type hinting

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager
    # Logger is inherited from BaseAgent.
except ImportError:
    # Fallback for direct script execution or import issues
    import sys
    import os
    import asyncio
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore


class DeepSeekAgent(BaseAgent):
    """
    An agent that utilizes DeepSeek AI models, often specialized for tasks like
    code generation, text completion, and complex problem-solving.
    It uses a chat-like interaction model via the APIManager.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DeepSeekAgent.

        Args:
            agent_name: The user-defined name for this agent instance.
            api_manager: An instance of `APIManager` for handling API calls to DeepSeek.
            config: Optional configuration dictionary for the agent.
                    Expected keys can include:
                    - "model" (str): The specific DeepSeek model to use (e.g., "deepseek-coder").
                    - "max_tokens" (int): Default maximum tokens for the response.
                    - "temperature" (float): Default sampling temperature.
                    - "default_system_prompt" (str): A default system prompt.
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager # Store APIManager for API calls
        
        # Set the DeepSeek model, defaulting if not specified in config.
        self.model: str = self.config.get("model", "deepseek-coder")
        self.logger.info(f"DeepSeekAgent '{self.agent_name}' initialized, configured for model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the DeepSeekAgent.

        Returns:
            A dictionary detailing the agent's purpose, primary skills (e.g., code generation),
            and example models it might support.
        """
        return {
            "description": "Agent for deep reasoning, complex problem solving, and code generation/analysis using DeepSeek AI models.",
            "capabilities": ["text_generation", "code_generation", "code_analysis", "complex_reasoning", "problem_solving"],
            "models_supported": ["deepseek-coder", "deepseek-llm"] # Example models, actual list may vary
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the DeepSeek API (typically a chat completions endpoint).

        Constructs a payload with system and user messages, then uses the `APIManager`
        to make an asynchronous API call. It parses the response to extract the
        generated content and other relevant information like finish reason and token usage.

        Args:
            query_data: A dictionary containing query details. Expected keys:
                        - "prompt" (str): The user's actual query or instruction. Mandatory.
                        - "system_prompt" (Optional[str]): A custom system prompt. If not provided,
                                                           the agent's default or a generic one is used.
                        - "max_tokens" (Optional[int]): Override default max tokens for this query.
                        - "temperature" (Optional[float]): Override default temperature for this query.
                        - Other API-specific parameters can also be included here if supported by APIManager.

        Returns:
            A dictionary with:
            - "status" (str): "success" or "error".
            - "content" (str, optional): The textual content of the AI's response.
            - "message" (str, optional): Error message if an error occurred.
            - "details" (Any, optional): Additional error details.
            - "finish_reason" (str, optional): Reason the model stopped generating.
            - "usage" (Dict, optional): Token usage data.
        """
        user_prompt = query_data.get("prompt")
        if not user_prompt: # Validate prompt presence
            self.logger.error("User prompt is missing in query_data for DeepSeekAgent.")
            return {"status": "error", "message": "User prompt missing"}

        # Determine the system prompt: query override > agent config default > generic default.
        default_system_prompt = self.config.get("default_system_prompt",
                                                "You are a helpful AI assistant specialized in coding and complex problem-solving. Provide clear and efficient solutions.")
        system_prompt_to_use = query_data.get("system_prompt", default_system_prompt)

        self.logger.info(f"Processing query for DeepSeekAgent '{self.agent_name}' using model '{self.model}'. Prompt (first 100 chars): '{user_prompt[:100]}...'")

        # Construct the payload for DeepSeek's chat completions format.
        messages: List[Dict[str,str]] = [
            {"role": "system", "content": system_prompt_to_use},
            {"role": "user", "content": user_prompt}
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 4096)),
            "temperature": query_data.get("temperature", self.config.get("temperature", 0.3)),
            # Other parameters like 'stream', 'top_p' can be added here from config or query_data
            # Example: "top_p": query_data.get("top_p", self.config.get("top_p", 0.9))
        }
        
        self.logger.debug(f"DeepSeek API request payload: {payload}")

        # Make the API call via APIManager.
        # 'deepseek' service must be configured in APIManager.
        # 'chat/completions' is a common endpoint for this type of model.
        response_data = await self.api_manager.make_request(
            service_name='deepseek',
            endpoint='chat/completions',
            method="POST",
            data=payload
        )

        # Handle errors from APIManager or the API itself.
        if response_data.get("error"):
            self.logger.error(f"API request to DeepSeek failed: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data)
            }

        # Parse the successful response from DeepSeek.
        try:
            # Typical DeepSeek/OpenAI-compatible API response structure:
            # { "choices": [ { "index": 0, "message": { "role": "assistant", "content": "response text" }, "finish_reason": "stop" } ], ... }
            choices = response_data.get("choices")
            if not choices or not isinstance(choices, list) or not choices[0]:
                self.logger.error(f"Invalid or empty 'choices' field in DeepSeek response. Response: {str(response_data)[:500]}")
                return {"status": "error", "message": "Invalid response structure from DeepSeek API (no choices)."}

            first_choice = choices[0]
            message_content = first_choice.get("message", {}).get("content")
            finish_reason = first_choice.get("finish_reason")
            
            if message_content is None: # Check if content is missing
                self.logger.error(f"No 'content' found in the first choice of DeepSeek response. Response: {str(response_data)[:500]}")
                return {"status": "error", "message": "No content in DeepSeek response message."}

            self.logger.info(f"Successfully received and parsed response from DeepSeek for prompt: '{user_prompt[:100]}...'")
            # Return structured success response
            return {
                "status": "success",
                "content": message_content,
                "finish_reason": finish_reason,
                "usage": response_data.get("usage") # Include token usage info if available
            }
        except (IndexError, KeyError, TypeError) as e: # Catch parsing errors
            self.logger.error(f"Error parsing DeepSeek response: {e}. Response data (first 500 chars): {str(response_data)[:500]}", exc_info=True)
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


        async def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            # Simulate async behavior if needed: await asyncio.sleep(0.01)
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

async def main_deepseek_test(): # Wrapped in async main function
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
    response1 = await deepseek_agent.process_query(query1_data) # Awaited
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
    response2 = await deepseek_agent.process_query(query2_data) # Awaited
    print(f"Response 2: {response2}")
    assert response2["status"] == "success"
    assert "def add_numbers" in response2.get("content", "") or "simulated response" in response2.get("content", "")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = await deepseek_agent.process_query(query3_data) # Awaited
    print(f"Response 3: {response3}")
    assert response3["status"] == "error"
    assert response3["message"] == "User prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = await deepseek_agent.process_query(query4_data) # Awaited
    print(f"Response 4: {response4}")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- DeepSeekAgent testing completed. ---")
    print("Note: The fallback import mechanism for BaseAgent/APIManager is primarily for isolated testing of this script.")
    print("In the full system, these should be resolved by Python's package structure.")

if __name__ == '__main__':
    # This block is for basic demonstration and testing.
    # It requires APIManager and BaseAgent to be correctly set up.

    # Setup a dummy APIManager and logger for local testing
    # In a real application, these would be part of the main system.
    from src.utils.logger import get_logger as setup_logger # type: ignore
    # import asyncio # Added at the top

    if os.name == 'nt': # Optional: Windows specific policy for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_deepseek_test())
