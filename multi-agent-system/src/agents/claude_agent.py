# src/agents/claude_agent.py
"""
Specialized agent for interacting with Anthropic's Claude AI models.

This agent handles communication with the Claude API via the APIManager,
formats requests according to Claude's specifications (e.g., messages API),
and processes the responses.
"""

import asyncio
from typing import Dict, Any, Optional, List # Added List for type hinting

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager
    # Logger is typically available via self.logger from BaseAgent, inherited from BaseAgent.
except ImportError:
    # Fallback for direct script execution or import issues (e.g., in testing scenarios)
    import sys
    import os
    # Ensure asyncio is available in fallback for the __main__ block test.
    import asyncio # Redundant if already imported, but safe.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore

class ClaudeAgent(BaseAgent):
    """
    An agent that utilizes Anthropic's Claude models for tasks such as
    text generation, summarization, question answering, and general analysis.
    It interacts with the Claude API using the messages format.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ClaudeAgent.

        Args:
            agent_name: The user-defined name for this agent instance.
            api_manager: An instance of `APIManager` to handle authenticated API calls to Claude.
            config: Optional configuration dictionary for the agent.
                    Expected keys might include:
                    - "model" (str): The specific Claude model to use (e.g., "claude-3-5-sonnet-20240620").
                    - "max_tokens" (int): Default maximum tokens for the response (Claude uses "max_tokens_to_sample").
                    - "max_tokens_to_sample" (int): Claude-specific max tokens parameter.
                    - "temperature" (float): Default sampling temperature.
                    - "default_system_prompt" (str): A default system prompt to use if none is provided in the query.
        """
        super().__init__(agent_name, config) # Initialize BaseAgent
        self.api_manager = api_manager # Store APIManager for making API calls
        
        # Set the Claude model to use, defaulting if not specified in config.
        self.model: str = self.config.get("model", "claude-3-5-sonnet-20240620")
        self.logger.info(f"ClaudeAgent '{self.agent_name}' initialized, configured for model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the ClaudeAgent.

        Returns:
            A dictionary outlining the agent's description, primary capabilities (tasks it can perform),
            and a list of Claude models it's known to support or is configured for.
        """
        return {
            "description": "Agent for general analysis, content creation, and conversational AI using Anthropic's Claude models.",
            "capabilities": ["text_generation", "summarization", "q&a", "general_analysis", "document_processing", "chat"],
            "models_supported": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-2.1", "claude-instant-1.2"]
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a given query by making a request to the Claude API.

        The method constructs a payload suitable for Claude's messages API,
        including the user's prompt and any specified system prompt. It then
        uses the `APIManager` to make the asynchronous API call and parses
        the response to extract the generated text content.

        Args:
            query_data: A dictionary containing the details for the query.
                        Expected keys:
                        - "prompt" (str): The user's actual query or message content. This is mandatory.
                        - "system_prompt" (Optional[str]): A system message to guide the AI's behavior.
                                                           If not provided, a default system prompt from the
                                                           agent's configuration or a generic one is used.
                        - "max_tokens_to_sample" (Optional[int]): Overrides the default max tokens for the response.
                        - "temperature" (Optional[float]): Overrides the default sampling temperature.

        Returns:
            A dictionary containing:
            - "status" (str): "success" or "error".
            - "content" (str, optional): The text content of Claude's response, if successful.
            - "message" (str, optional): An error message, if an error occurred.
            - "details" (Any, optional): Additional details from the API response or error.
            - "finish_reason" (str, optional): The reason the model stopped generating text (e.g., "end_turn", "max_tokens").
            - "usage" (Dict, optional): Token usage information (e.g., {"input_tokens": ..., "output_tokens": ...}).
        """
        user_prompt = query_data.get("prompt")
        if not user_prompt: # Validate that a prompt is provided
            self.logger.error("User prompt is missing in query_data for ClaudeAgent.")
            return {"status": "error", "message": "User prompt missing"}

        # Determine the system prompt to use.
        # Priority: query_data override > agent config default > generic default.
        system_prompt_override = query_data.get("system_prompt")
        if system_prompt_override is None: # Only use default if 'system_prompt' key is not in query_data at all
            system_prompt_to_use = self.config.get("default_system_prompt", "You are a helpful AI assistant.")
        else: # If 'system_prompt' is present (even if an empty string), use it.
              # Claude API handles empty system prompts by not using one.
            system_prompt_to_use = system_prompt_override

        self.logger.info(f"Processing query for ClaudeAgent '{self.agent_name}' with model '{self.model}'. Prompt (first 100 chars): '{user_prompt[:100]}...'")

        # Construct the messages list for the Claude API (currently only the user prompt)
        messages: List[Dict[str, str]] = [{"role": "user", "content": user_prompt}]
        
        # Prepare the payload for the Claude API request
        # Uses max_tokens_to_sample (Claude specific) or falls back to max_tokens from config.
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": query_data.get("max_tokens_to_sample") or \
                          self.config.get("max_tokens_to_sample") or \
                          self.config.get("max_tokens", 2000), # Default if no other max_tokens found
            "temperature": query_data.get("temperature", self.config.get("temperature", 0.5)),
            # Other Claude-specific parameters like "stream", "top_k", "top_p" can be added here from query_data or config
        }

        # Add system prompt to payload only if it's non-empty.
        if system_prompt_to_use:
            payload["system"] = system_prompt_to_use
        
        self.logger.debug(f"Claude API request payload: {payload}")

        # Make the API call via APIManager.
        # The endpoint for Claude messages API is typically "/v1/messages".
        response_data = await self.api_manager.make_request(
            service_name='claude', # Service name must match a configuration in APIManager
            endpoint='messages',   # Standard endpoint for Claude's messages API
            method="POST",
            data=payload           # The constructed payload
        )

        # Handle potential errors from the APIManager (e.g., network issues, HTTP errors)
        if response_data.get("error"):
            self.logger.error(f"API request to Claude failed: {response_data.get('message', response_data.get('error'))}")
            return { # Propagate a structured error
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data) # Include error content if available
            }

        # Try to parse the successful response from Claude
        try:
            # Expected Claude messages API response structure:
            # { "id": ..., "type": "message", "role": "assistant",
            #   "content": [ { "type": "text", "text": "Actual response text here..." } ], ... }
            content_list = response_data.get("content")
            if not content_list or not isinstance(content_list, list) or not content_list:
                self.logger.error(f"Unexpected response structure from Claude (no content list or empty). Response: {str(response_data)[:500]}")
                return {"status": "error", "message": "Invalid response structure from Claude API (missing or empty 'content' list)."}

            extracted_text = ""
            # Iterate through content blocks; typically one text block for simple text prompts.
            for block in content_list:
                if isinstance(block, dict) and block.get("type") == "text":
                    extracted_text += block.get("text", "") # Append text from each block
            
            if not extracted_text: # If, after iterating, no text was extracted
                self.logger.error(f"No text found in Claude response content blocks. Response: {str(response_data)[:500]}")
                return {"status": "error", "message": "No text content found in Claude's response."}

            self.logger.info(f"Successfully received and parsed response from Claude for prompt: '{user_prompt[:100]}...'")
            # Return a structured success response
            return {
                "status": "success",
                "content": extracted_text,
                "finish_reason": response_data.get("stop_reason"), # e.g., "end_turn", "max_tokens"
                "usage": response_data.get("usage") # e.g., {"input_tokens": ..., "output_tokens": ...}
            }
        except (IndexError, KeyError, TypeError) as e: # Catch errors during response parsing
            self.logger.error(f"Error parsing Claude response: {e}. Response data (first 500 chars): {str(response_data)[:500]}", exc_info=True)
            return {"status": "error", "message": f"Error parsing Claude response: {e}"}

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore

    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_ClaudeTest")
            self.service_configs = {
                "claude": {
                    "api_key": "dummy_claude_key",
                    "base_url": "https://api.anthropic.com/v1" # Not actually used by dummy
                }
            }

        async def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            # Simulate some async behavior if needed, e.g., await asyncio.sleep(0.01)
            if service_name == "claude" and endpoint == "messages":
                if "error" in data.get("messages")[0].get("content","").lower(): # Simulate error
                     return {"error": "Simulated API Error", "message": "The prompt contained 'error'", "status_code": 400}

                response_text = f"This is a simulated Claude response to: '{data['messages'][0]['content']}' with model {data.get('model')}."
                if data.get("system"):
                    response_text += f" System prompt was: '{data['system']}'."
                
                return {
                    "id": "msg_xxxxxxxxxxxxxxxxx",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ],
                    "model": data.get("model"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": sum(len(m["content"]) for m in data["messages"]), # Simplified
                        "output_tokens": len(response_text)
                    }
                }
            return {"error": "Unknown service or endpoint in DummyAPIManager", "status_code": 404}

async def main_claude_test(): # Wrapped in async main function
    print("--- Testing ClaudeAgent ---")
    
    dummy_api_manager = DummyAPIManager()
    agent_config = {
        "model": "claude-3-opus-test",
        "max_tokens": 150, # Claude API might use max_tokens_to_sample
        "max_tokens_to_sample": 150, 
        "temperature": 0.6,
        "default_system_prompt": "You are a specialized Claude test assistant."
    }
    
    claude_agent = ClaudeAgent(
        agent_name="TestClaudeAgent001",
        api_manager=dummy_api_manager, # type: ignore
        config=agent_config
    )

    print(f"Agent Name: {claude_agent.get_name()}")
    print(f"Agent Capabilities: {claude_agent.get_capabilities()}")
    print(f"Agent Configured Model: {claude_agent.model}")

    # Test case 1: Simple query using default system prompt
    print("\n--- Test Case 1: Simple Query (Default System Prompt) ---")
    query1_data = {"prompt": "Explain black holes in simple terms."}
    response1 = await claude_agent.process_query(query1_data) # Awaited
    print(f"Response 1: {response1}")
    assert response1["status"] == "success"
    assert "simulated Claude response" in response1.get("content", "")
    assert agent_config["default_system_prompt"] in response1.get("content", "")


    # Test case 2: Query with custom system prompt
    print("\n--- Test Case 2: Custom System Prompt ---")
    custom_sys_prompt = "You are a historian specializing in ancient Rome."
    query2_data = {
        "prompt": "Tell me about Julius Caesar.",
        "system_prompt": custom_sys_prompt
    }
    response2 = await claude_agent.process_query(query2_data) # Awaited
    print(f"Response 2: {response2}")
    assert response2["status"] == "success"
    assert custom_sys_prompt in response2.get("content", "")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = await claude_agent.process_query(query3_data) # Awaited
    print(f"Response 3: {response3}")
    assert response3["status"] == "error"
    assert response3["message"] == "User prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = await claude_agent.process_query(query4_data) # Awaited
    print(f"Response 4: {response4}")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    # Test case 5: Empty string as system_prompt (should not use default)
    print("\n--- Test Case 5: Empty String System Prompt ---")
    query5_data = {
        "prompt": "What is the weather like today?",
        "system_prompt": "" 
    }
    response5 = await claude_agent.process_query(query5_data) # Awaited
    print(f"Response 5: {response5}")
    assert response5["status"] == "success"
    assert agent_config["default_system_prompt"] not in response5.get("content", "") # Ensure default was NOT used
    assert "System prompt was: ''" in response5.get("content", "") or "System prompt was" not in response5.get("content", "") # Depending on dummy impl.

    print("\n--- ClaudeAgent testing completed. ---")
    print("Note: The fallback import mechanism is primarily for isolated testing.")
    print("In the full system, imports should be resolved by Python's package structure.")

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore
    # Ensure asyncio is imported, typically at the top of the file.
    # import asyncio # Already added at the top

    # Optional: Windows specific policy for asyncio if needed for tests
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_claude_test())
