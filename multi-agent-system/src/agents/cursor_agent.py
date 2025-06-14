# src/agents/cursor_agent.py
"""
Specialized agent for interacting with a hypothetical Cursor AI,
focusing on code generation, editing, and other programming-related tasks.

Note: As the actual Cursor API details are not publicly available (as of the
time of this writing), this agent is based on a speculative API structure.
Its functionality is for demonstration and would require significant adjustments
to work with a real Cursor API.
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

class CursorAgent(BaseAgent):
    """
    An agent designed to utilize a hypothetical Cursor AI for tasks like
    code generation, refactoring, and other programming-related activities.

    The implementation assumes a generic API interaction pattern where prompts
    and parameters are sent to a Cursor endpoint, and a textual or structured
    code response is received.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CursorAgent.

        Args:
            agent_name: The user-defined name for this agent instance.
            api_manager: An instance of `APIManager` to handle API calls.
            config: Optional configuration dictionary for the agent.
                    Expected keys might include:
                    - "model" (str): Specific Cursor model/version to use (e.g., "cursor-pro").
                    - "mode" (str): Default operational mode (e.g., "code-generation", "edit-code", "chat").
                    - "default_system_prompt" (str): A default system message for guiding the AI.
                    - "max_tokens" (int): Default maximum tokens for the response.
                    - "temperature" (float): Default sampling temperature.
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager # For making API calls
        
        # Set default operational mode and model from config, or use placeholders.
        self.model_mode: str = self.config.get("mode", "code-generation")
        self.model: str = self.config.get("model", "cursor-default") # Hypothetical model name
        self.default_system_prompt: str = self.config.get(
            "default_system_prompt", 
            "You are an AI programming assistant. Follow the user's requirements carefully and to the letter. Prioritize correctness and clarity in your code."
        )
        self.logger.info(f"CursorAgent '{self.agent_name}' initialized. Default mode: '{self.model_mode}', Model: '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the CursorAgent.

        Returns:
            A dictionary outlining the agent's description, primary capabilities
            (e.g., code generation, refactoring), and example supported modes.
        """
        return {
            "description": "Agent for code generation, refactoring, and programming tasks using a hypothetical Cursor AI.",
            "capabilities": ["code_generation", "refactoring", "code_completion", "code_analysis", "code_editing", "explain_code"],
            "modes_supported": ["code-generation", "edit-code", "chat", "explain_code"] # Example modes
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the (hypothetical) Cursor API.

        Constructs a payload based on the query data, including prompt, system prompt,
        mode, and other potential Cursor-specific parameters. It then uses the
        `APIManager` to make the API call and parses the response.

        Args:
            query_data: A dictionary containing the query details. Expected keys:
                        - "prompt" (str): The user's main query, code snippet, or instruction. Mandatory.
                        - "system_prompt" (Optional[str]): A custom system prompt to override the default.
                        - "mode" (Optional[str]): Overrides the agent's default operational mode for this query.
                        - "code_context" (Optional[str]): Existing code to provide context for generation or editing.
                        - "file_path" (Optional[str]): Path to the relevant file, if applicable.
                        - "selection" (Optional[Dict]): Information about selected code (e.g., line numbers).
                        - "max_tokens" (Optional[int]): Override default max tokens for the response.
                        - "temperature_override" (Optional[float]): Override default temperature.
        Returns:
            A dictionary containing:
            - "status" (str): "success" or "error".
            - "content" (str, optional): The primary textual or code content of the response.
            - "message" (str, optional): An error message, if an error occurred.
            - "details" (Any, optional): Additional details from the API error response.
            - "raw_response" (Dict, optional): The full, raw response from the API for debugging.
        """
        user_query = query_data.get("prompt")
        if not user_query: # A prompt or some primary input is essential
            self.logger.error("User query/prompt is missing in query_data for CursorAgent.")
            return {"status": "error", "message": "User query/prompt missing"}

        # Determine system prompt and operational mode, using overrides or defaults.
        system_prompt_to_use = query_data.get("system_prompt", self.default_system_prompt)
        current_mode = query_data.get("mode", self.model_mode)

        self.logger.info(f"Processing query for CursorAgent '{self.agent_name}' (mode: {current_mode}). Query (first 100 chars): '{user_query[:100]}...'")

        # Construct the full prompt for the Cursor API.
        # This is speculative; a real API might prefer a structured "messages" array.
        # For now, we combine system prompt, user query, and any code context into a single string.
        full_prompt_parts: List[str] = []
        if system_prompt_to_use: # Only add system prompt if it's non-empty
            full_prompt_parts.append(system_prompt_to_use)
        full_prompt_parts.append(f"User Query:\n{user_query}")

        if query_data.get("code_context"): # Append existing code context if provided
            full_prompt_parts.append(f"\n\nExisting Code Context:\n{query_data.get('code_context')}")

        full_prompt = "\n\n".join(full_prompt_parts)

        # Prepare the payload for the Cursor API.
        payload: Dict[str, Any] = {
            "prompt": full_prompt,    # The combined prompt text.
            "model": self.model,      # Specified model for Cursor.
            "mode": current_mode,     # Operational mode (e.g., generate, edit).
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 2048)), # Max response tokens.
        }
        
        # Apply temperature settings (config default or query-time override).
        # Temperature controls the randomness/creativity of the output.
        temperature_override = query_data.get("temperature_override")
        if temperature_override is not None:
            payload["temperature"] = temperature_override
        elif self.config.get("temperature") is not None:
            payload["temperature"] = self.config.get("temperature")

        # Add any other hypothetical Cursor-specific parameters from query_data.
        if query_data.get("file_path"):
            payload["file_path"] = query_data.get("file_path")
        if query_data.get("selection"): # E.g., for code editing context.
            payload["selection"] = query_data.get("selection")

        self.logger.debug(f"Cursor API request payload: {payload}")

        # Make the API call using APIManager.
        # 'cursor' is the service name configured in APIManager.
        # 'compose' is a hypothetical endpoint for Cursor (e.g., /v1/compose or /v1/generate).
        response_data = await self.api_manager.make_request(
            service_name='cursor', 
            endpoint='compose',
            method="POST",
            data=payload
        )

        # Handle errors returned by APIManager (e.g., network, HTTP status codes).
        if response_data.get("error"):
            self.logger.error(f"API request to Cursor failed: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data) # Include error details if available
            }

        # Try to parse the successful response from Cursor.
        # This part is highly speculative due to the unknown nature of a real Cursor API.
        try:
            # Attempt to extract content from common possible fields in an API response.
            extracted_content = response_data.get("response", response_data.get("generated_code"))
            if extracted_content is None: # Fallback to other common fields
                extracted_content = response_data.get("text", response_data.get("completion"))

            if extracted_content is None: # If no content found after checking common fields
                self.logger.error(f"Failed to extract meaningful content from Cursor response. Response (first 300 chars): {str(response_data)[:300]}")
                return {"status": "error", "message": "Invalid or empty response structure from Cursor API."}

            self.logger.info(f"Successfully received and parsed response from Cursor for query: '{user_query[:100]}...'")
            return {
                "status": "success",
                "content": extracted_content, # The primary content (e.g., generated code, explanation)
                "raw_response": response_data # Optionally include the full raw API response for debugging
            }
        except Exception as e: # Broad exception for unexpected issues during parsing
            self.logger.error(f"Error parsing Cursor response: {e}. Response data: {str(response_data)[:500]}", exc_info=True)
            return {"status": "error", "message": f"Error parsing Cursor response: {e}"}

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore

    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_CursorTest")
            self.service_configs = { # Needed for APIManager.get_auth_header, etc.
                "cursor": {"api_key": "dummy_cursor_key", "base_url": "https://api.cursor.ai/v1"}
            }

        async def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            # Simulate async behavior if needed: await asyncio.sleep(0.01)
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

async def main_cursor_test(): # Wrapped in async main function
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
    response1 = await cursor_agent.process_query(query1_data) # Awaited
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
    response2 = await cursor_agent.process_query(query2_data) # Awaited
    print(f"Response 2:\n{response2.get('content')}\n")
    assert response2["status"] == "success"
    assert "edit-code" in response2.get("content", "") # Dummy response indicates mode
    assert custom_sys_prompt in response2.get("raw_response",{}).get("response","") # Check if system prompt was part of input to dummy

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = await cursor_agent.process_query(query3_data) # Awaited
    print(f"Response 3: {response3}\n")
    assert response3["status"] == "error"
    assert response3["message"] == "User query/prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error."} # DummyAPIManager will simulate error
    response4 = await cursor_agent.process_query(query4_data) # Awaited
    print(f"Response 4: {response4}\n")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- CursorAgent testing completed. ---")
    print("Note: This agent's implementation is highly speculative due to lack of public Cursor API details.")
    print("The API interaction (payload, endpoint, response parsing) will likely need significant adjustments if a real API is available.")

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore
    # import asyncio # Added at the top

    if os.name == 'nt': # Optional: Windows specific policy for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_cursor_test())
