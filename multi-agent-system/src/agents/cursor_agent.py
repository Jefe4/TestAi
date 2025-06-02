# src/agents/cursor_agent.py
"""Specialized agent for interacting with Cursor AI, focusing on code generation and editing."""

from typing import Dict, Any, Optional

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager
except ImportError:
    # Fallback for direct script execution or import issues
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore

class CursorAgent(BaseAgent):
    """
    An agent that utilizes Cursor AI for tasks like code generation,
    refactoring, and other programming-related activities.
    Note: The actual Cursor API details are speculative as of this implementation.
    This agent is based on a hypothetical API structure.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CursorAgent.

        Args:
            agent_name: The name of the agent.
            api_manager: An instance of APIManager to handle API calls.
            config: Optional configuration dictionary for the agent.
                    Expected keys: "mode", "default_system_prompt", "model" (optional),
                                   "max_tokens", "temperature".
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager

        self.model_mode = self.config.get("mode", "code-generation") # e.g., "code-generation", "edit-code", "chat"
        self.model = self.config.get("model", "cursor-default") # Hypothetical model name
        self.default_system_prompt = self.config.get(
            "default_system_prompt",
            "You are an AI programming assistant. Follow the user's requirements carefully and to the letter."
        )
        self.logger.info(f"CursorAgent '{self.agent_name}' initialized with mode '{self.model_mode}' and model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the CursorAgent.
        """
        return {
            "description": "Agent for code generation, refactoring, and programming tasks using a hypothetical Cursor AI.",
            "capabilities": ["code_generation", "refactoring", "code_completion", "code_analysis", "code_editing"],
            "modes_supported": ["code-generation", "edit-code", "chat"] # Example modes
        }

    def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the (hypothetical) Cursor API.

        Args:
            query_data: A dictionary containing the query details.
                        Expected keys:
                        - "prompt" (str): The user's actual query or code context.
                        - "system_prompt" (Optional[str]): Custom system prompt for this query.
                        - "mode" (Optional[str]): Override the agent's default mode.
                        - Other potential keys: "code_to_edit", "cursor_position", etc.

        Returns:
            A dictionary containing the status of the operation and the response content or error message.
        """
        user_query = query_data.get("prompt")
        if not user_query: # A prompt or some form of input is essential
            self.logger.error("User query/prompt is missing in query_data.")
            return {"status": "error", "message": "User query/prompt missing"}

        system_prompt_override = query_data.get("system_prompt")
        system_prompt_to_use = system_prompt_override if system_prompt_override is not None else self.default_system_prompt

        current_mode = query_data.get("mode", self.model_mode)

        self.logger.info(f"Processing query for CursorAgent '{self.agent_name}' (mode: {current_mode}) with query: '{user_query[:100]}...'")

        # Constructing the prompt for Cursor.
        # This is speculative. A real Cursor API might have structured inputs for messages, code, etc.
        # For now, we'll assume a combined text prompt for simplicity, similar to some older APIs.
        # Or, if the API expects a messages array like OpenAI:
        # messages = [
        #    {"role": "system", "content": system_prompt_to_use},
        #    {"role": "user", "content": user_query}
        # ]
        # payload["messages"] = messages
        # For now, using a single "prompt" field with combined text:

        # If the API is more like OpenAI/Claude with a messages structure:
        # messages = [{"role": "system", "content": system_prompt_to_use}, {"role": "user", "content": user_query}]
        # payload = {"messages": messages, "model": self.model, ...}
        # However, the original subtask description implied a single "prompt" field.
        # Let's stick to that for now and assume the API handles it.
        full_prompt = f"{system_prompt_to_use}\n\nUser Query:\n{user_query}"
        if query_data.get("code_context"): # Example of adding more context
            full_prompt += f"\n\nExisting Code Context:\n{query_data.get('code_context')}"


        payload: Dict[str, Any] = {
            "prompt": full_prompt, # This is based on the subtask's structure.
            "model": self.model, # Model to use, if applicable to Cursor API
            "mode": current_mode,
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 2048)),
        }

        if self.config.get("temperature") is not None:
            payload["temperature"] = self.config.get("temperature")
        if query_data.get("temperature_override") is not None: # Allow query-time override
            payload["temperature"] = query_data.get("temperature_override")

        # Cursor specific parameters (hypothetical)
        if query_data.get("file_path"):
            payload["file_path"] = query_data.get("file_path")
        if query_data.get("selection"): # e.g., line numbers or character offsets for edits
            payload["selection"] = query_data.get("selection")


        self.logger.debug(f"Cursor API payload: {payload}")

        # Make the API call via APIManager. Endpoint 'compose' or similar.
        # The service name 'cursor' must be configured in APIManager.
        response_data = self.api_manager.make_request(
            service_name='cursor',
            endpoint='compose', # Hypothetical endpoint, e.g., /v1/compose
            method="POST",
            data=payload
        )

        if response_data.get("error"):
            self.logger.error(f"API request failed for Cursor: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data)
            }

        try:
            # Assuming Cursor API response might have a "response" or "generated_code" field.
            # This is highly speculative.
            extracted_content = response_data.get("response", response_data.get("generated_code"))
            if extracted_content is None: # Check for other common fields if primary ones are missing
                extracted_content = response_data.get("text", response_data.get("completion"))

            if extracted_content is None:
                self.logger.error(f"Failed to extract content from Cursor response. Response: {response_data}")
                return {"status": "error", "message": "Invalid response structure from Cursor API."}

            self.logger.info(f"Successfully received and parsed response from Cursor for query: '{user_query[:100]}...'")
            return {
                "status": "success",
                "content": extracted_content,
                "raw_response": response_data # Optionally include the full response
            }
        except Exception as e: # Broad exception for parsing if structure is unknown
            self.logger.error(f"Error parsing Cursor response: {e}. Response data: {response_data}")
            return {"status": "error", "message": f"Error parsing Cursor response: {e}"}

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore

    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_CursorTest")
            self.service_configs = { # Needed for APIManager.get_auth_header, etc.
                "cursor": {"api_key": "dummy_cursor_key", "base_url": "https://api.cursor.ai/v1"}
            }

        def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            if service_name == "cursor" and endpoint == "compose":
                if "error" in data.get("prompt","").lower():
                     return {"error": "Simulated API Error", "message": "The prompt contained 'error'", "status_code": 400}

                # Simulate response based on mode
                mode = data.get("mode", "code-generation")
                response_text = f"Simulated Cursor response for mode '{mode}'. Query was: '{data.get('prompt', '')[:100]}...'"
                if mode == "code-generation":
                    response_text += "\n```python\n# Sample generated code\nprint('Hello from Cursor!')\n```"
                elif mode == "edit-code":
                     response_text += "\n```python\n# Original code would be here, with edits applied\nprint('Hello from edited Cursor code!')\n```"

                return {
                    "id": "curs_xxxxxxxxxxxxxxxxx",
                    "response": response_text, # Main field for generated content
                    "generated_code": response_text if "code" in mode else None, # More specific field
                    "model_used": data.get("model", "cursor-default-simulated"),
                    "usage": {"prompt_tokens": len(data.get("prompt","")), "completion_tokens": len(response_text)}
                }
            return {"error": "Unknown service or endpoint in DummyAPIManager", "status_code": 404}

    print("--- Testing CursorAgent ---")

    dummy_api_manager = DummyAPIManager()
    agent_config = {
        "model": "cursor-test-model",
        "mode": "code-generation", # Default mode for this agent instance
        "max_tokens": 1024,
        "temperature": 0.2,
        "default_system_prompt": "You are an expert Cursor AI. Generate precise code."
    }

    cursor_agent = CursorAgent(
        agent_name="TestCursorAgent001",
        api_manager=dummy_api_manager, # type: ignore
        config=agent_config
    )

    print(f"Agent Name: {cursor_agent.get_name()}")
    print(f"Agent Capabilities: {cursor_agent.get_capabilities()}")
    print(f"Agent Configured Model: {cursor_agent.model}, Mode: {cursor_agent.model_mode}")

    # Test case 1: Code generation query
    print("\n--- Test Case 1: Code Generation Query ---")
    query1_data = {"prompt": "Generate a Python function to calculate factorial."}
    response1 = cursor_agent.process_query(query1_data)
    print(f"Response 1:\n{response1.get('content')}\n") # Print content for readability
    assert response1["status"] == "success"
    assert "factorial" in response1.get("content", "") or "simulated Cursor response" in response1.get("content", "")

    # Test case 2: Query with custom system prompt and mode override
    print("\n--- Test Case 2: Custom System Prompt & Mode Override ---")
    custom_sys_prompt = "You are a code refactoring expert. Suggest improvements."
    query2_data = {
        "prompt": "Refactor this Python snippet: for i in range(len(my_list)): print(my_list[i])",
        "system_prompt": custom_sys_prompt,
        "mode": "edit-code" # Override agent's default mode
    }
    response2 = cursor_agent.process_query(query2_data)
    print(f"Response 2:\n{response2.get('content')}\n")
    assert response2["status"] == "success"
    assert "edit-code" in response2.get("content", "") # Dummy response indicates mode
    assert custom_sys_prompt in response2.get("raw_response",{}).get("response","") # Check if system prompt was part of input to dummy

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = cursor_agent.process_query(query3_data)
    print(f"Response 3: {response3}\n")
    assert response3["status"] == "error"
    assert response3["message"] == "User query/prompt missing"

    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = cursor_agent.process_query(query4_data)
    print(f"Response 4: {response4}\n")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- CursorAgent testing completed. ---")
    print("Note: This agent's implementation is highly speculative due to lack of public Cursor API details.")
    print("The API interaction (payload, endpoint, response parsing) will likely need significant adjustments if a real API is available.")
