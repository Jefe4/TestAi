# src/agents/claude_agent.py
"""Specialized agent for interacting with Anthropic's Claude AI models."""

from typing import Dict, Any, Optional

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager
    # Logger is typically available via self.logger from BaseAgent
except ImportError:
    # Fallback for direct script execution or import issues
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore

class ClaudeAgent(BaseAgent):
    """
    An agent that utilizes Anthropic's Claude models for tasks like
    text generation, summarization, Q&A, and general analysis.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the ClaudeAgent.

        Args:
            agent_name: The name of the agent.
            api_manager: An instance of APIManager to handle API calls.
            config: Optional configuration dictionary for the agent.
                    Expected keys: "model", "max_tokens", "temperature", "default_system_prompt".
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager
        
        # Default model if not specified in config
        self.model = self.config.get("model", "claude-3-5-sonnet-20240620") 
        self.logger.info(f"ClaudeAgent '{self.agent_name}' initialized with model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the ClaudeAgent.
        """
        return {
            "description": "Agent for general analysis, content creation, and conversational AI using Claude.",
            "capabilities": ["text_generation", "summarization", "q&a", "general_analysis", "document_processing"],
            "models_supported": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-haiku-20240307"] 
        }

    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the Claude API.

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

        system_prompt_override = query_data.get("system_prompt")
        # Use agent's default system prompt if no override is provided in query_data
        # If system_prompt_override is an empty string, it will be passed as such to Claude.
        # Claude API handles empty system prompts by not using one, which is fine.
        if system_prompt_override is None: # only if not present in query_data
            system_prompt_to_use = self.config.get("default_system_prompt", "You are a helpful AI assistant.")
        else:
            system_prompt_to_use = system_prompt_override


        self.logger.info(f"Processing query for ClaudeAgent '{self.agent_name}' with prompt: '{user_prompt[:100]}...'")

        messages = [{"role": "user", "content": user_prompt}]
        
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": query_data.get("max_tokens_to_sample", # Claude uses max_tokens_to_sample
                                         self.config.get("max_tokens", # Fallback to generic max_tokens
                                                         self.config.get("max_tokens_to_sample", 2000))), # Then specific
            "temperature": query_data.get("temperature", self.config.get("temperature", 0.5)),
            # Other Claude-specific parameters like "stream", "top_k", "top_p" can be added here
        }

        if system_prompt_to_use: # Only add system prompt to payload if it's not empty or None
            payload["system"] = system_prompt_to_use
        
        self.logger.debug(f"Claude API payload: {payload}")

        # Make the API call via APIManager
        # Endpoint for Claude messages API is typically /v1/messages
        response_data = self.api_manager.make_request(
            service_name='claude',
            endpoint='messages', # APIManager will prepend base_url (e.g., https://api.anthropic.com/v1)
            method="POST",
            data=payload
        )

        if response_data.get("error"):
            self.logger.error(f"API request failed for Claude: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data)
            }

        try:
            # Claude's response structure for messages API:
            # { "content": [ { "type": "text", "text": "response text" } ], "role": "assistant", ... }
            if not response_data.get("content") or not isinstance(response_data["content"], list) or not response_data["content"]:
                self.logger.error(f"Unexpected response structure from Claude (no content list). Response: {response_data}")
                return {"status": "error", "message": "Invalid response structure from Claude API (no content)."}

            extracted_text = ""
            # Iterate through content blocks, though for simple text prompts, usually one text block.
            for block in response_data["content"]:
                if block.get("type") == "text":
                    extracted_text += block.get("text", "")
            
            if not extracted_text: # If loop finishes and text is still empty
                self.logger.error(f"No text found in Claude response content blocks. Response: {response_data}")
                return {"status": "error", "message": "No text content found in Claude response."}

            self.logger.info(f"Successfully received and parsed response from Claude for prompt: '{user_prompt[:100]}...'")
            return {
                "status": "success",
                "content": extracted_text,
                "finish_reason": response_data.get("stop_reason"),
                "usage": response_data.get("usage") # { "input_tokens": ..., "output_tokens": ... }
            }
        except (IndexError, KeyError, TypeError) as e:
            self.logger.error(f"Error parsing Claude response: {e}. Response data: {response_data}")
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

        def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
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
    response1 = claude_agent.process_query(query1_data)
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
    response2 = claude_agent.process_query(query2_data)
    print(f"Response 2: {response2}")
    assert response2["status"] == "success"
    assert custom_sys_prompt in response2.get("content", "")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = claude_agent.process_query(query3_data)
    print(f"Response 3: {response3}")
    assert response3["status"] == "error"
    assert response3["message"] == "User prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = claude_agent.process_query(query4_data)
    print(f"Response 4: {response4}")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    # Test case 5: Empty string as system_prompt (should not use default)
    print("\n--- Test Case 5: Empty String System Prompt ---")
    query5_data = {
        "prompt": "What is the weather like today?",
        "system_prompt": "" 
    }
    response5 = claude_agent.process_query(query5_data)
    print(f"Response 5: {response5}")
    assert response5["status"] == "success"
    assert agent_config["default_system_prompt"] not in response5.get("content", "") # Ensure default was NOT used
    assert "System prompt was: ''" in response5.get("content", "") or "System prompt was" not in response5.get("content", "") # Depending on dummy impl.

    print("\n--- ClaudeAgent testing completed. ---")
    print("Note: The fallback import mechanism is primarily for isolated testing.")
    print("In the full system, imports should be resolved by Python's package structure.")
