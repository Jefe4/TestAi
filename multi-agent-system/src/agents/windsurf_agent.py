# src/agents/windsurf_agent.py
"""
Specialized agent for interacting with a hypothetical Windsurf AI,
focusing on web development, UI/UX, frontend frameworks, and CSS.

Note: The Windsurf AI and its API are purely speculative for the purpose
of demonstrating a specialized agent. This implementation would need to be
adapted to a real API if one existed.
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

class WindsurfAgent(BaseAgent):
    """
    An agent that interfaces with a hypothetical "Windsurf AI" service,
    specializing in web development topics such as UI/UX design principles,
    frontend framework advice (React, Vue, Angular), CSS styling techniques,
    and accessibility best practices.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the WindsurfAgent.

        Args:
            agent_name: The user-defined name for this agent instance.
            api_manager: An instance of `APIManager` for handling API calls.
            config: Optional configuration dictionary. Expected keys might include:
                    - "model" (str): Specific Windsurf AI model/version (e.g., "windsurf-web-expert-v1").
                    - "focus" (str): Default focus area for queries (e.g., "react", "css-architecture").
                    - "default_system_prompt" (str): A default system message for the AI.
                    - "max_tokens" (int): Default maximum tokens for API responses.
                    - "temperature" (float): Default sampling temperature.
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager # For making API calls
        
        # Set default focus area, model, and system prompt from config, or use placeholders.
        self.focus_area: str = self.config.get("focus", "web-development")
        self.model: str = self.config.get("model", "windsurf-latest") # Hypothetical model name
        self.default_system_prompt: str = self.config.get(
            "default_system_prompt", 
            "You are a web development expert. Provide clear, actionable solutions and best practices related to frontend frameworks, CSS, UI/UX, and modern web development standards."
        )
        self.logger.info(f"WindsurfAgent '{self.agent_name}' initialized. Default focus: '{self.focus_area}', Model: '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the WindsurfAgent.

        Returns:
            A dictionary detailing the agent's specialization, primary skills,
            and example focus areas it might support.
        """
        return {
            "description": "Agent specializing in web development topics including UI/UX, frontend frameworks (React, Vue, Angular), CSS, and accessibility, using a hypothetical Windsurf AI.",
            "capabilities": ["web_development_advice", "ui_ux_principles", "frontend_framework_guidance", "css_styling_techniques", "accessibility_best_practices", "code_examples_web"],
            "focus_areas_supported": ["general_web_dev", "react", "vue", "angular", "css_grid_flexbox", "web_performance", "pwa", "web_apis"] # Example focus areas
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the hypothetical Windsurf API.

        This method constructs a payload tailored for a web development AI,
        including the user's prompt, system prompt, focus area, and other
        relevant parameters. It then uses the `APIManager` to make the API request.

        Args:
            query_data: A dictionary containing query details. Expected keys:
                        - "prompt" (str): The user's question or task description. Mandatory.
                        - "system_prompt" (Optional[str]): Custom system prompt to override the default.
                        - "focus" (Optional[str]): Specific focus area for this query (e.g., "react", "css").
                        - "context_details" (Optional[str]): Additional context for the query (e.g., project requirements).
                        - "output_format" (Optional[str]): Desired output format (e.g., "html", "css", "javascript", "explanation").
                        - "framework_version" (Optional[str]): Specific framework version if relevant (e.g., "React 18").
                        - "max_tokens" (Optional[int]): Override default max tokens for the response.
                        - "temperature_override" (Optional[float]): Override default temperature.
        Returns:
            A dictionary with:
            - "status" (str): "success" or "error".
            - "content" (str, optional): The textual or code content of the AI's response.
            - "message" (str, optional): Error message if an error occurred.
            - "details" (Any, optional): Additional error details.
            - "raw_response" (Dict, optional): The full, raw response from the API.
        """
        user_query = query_data.get("prompt")
        if not user_query: # Validate prompt
            self.logger.error("User query/prompt is missing in query_data for WindsurfAgent.")
            return {"status": "error", "message": "User query/prompt missing"}

        # Determine system prompt and focus area, using overrides or agent defaults.
        system_prompt_to_use = query_data.get("system_prompt", self.default_system_prompt)
        current_focus = query_data.get("focus", self.focus_area)

        self.logger.info(f"Processing query for WindsurfAgent '{self.agent_name}' (Focus: {current_focus}). Query (first 100 chars): '{user_query[:100]}...'")

        # Construct the full prompt. A real API might use a more structured format (e.g., messages array).
        full_prompt_parts: List[str] = []
        if system_prompt_to_use: # Add system prompt if defined
            full_prompt_parts.append(f"System Prompt:\n{system_prompt_to_use}")
        full_prompt_parts.append(f"User Query:\n{user_query}")

        if query_data.get("context_details"): # Append additional context if provided
            full_prompt_parts.append(f"\n\nAdditional Context:\n{query_data.get('context_details')}")

        full_prompt = "\n\n".join(full_prompt_parts)

        # Prepare the payload for the Windsurf API.
        payload: Dict[str, Any] = {
            "prompt": full_prompt,
            "model": self.model,        # Specified model for Windsurf AI.
            "focus": current_focus,     # Current focus area (e.g., "react", "css").
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 2000)), # Max response tokens.
        }
        
        # Apply temperature settings from query or config.
        temperature_override = query_data.get("temperature_override")
        if temperature_override is not None:
            payload["temperature"] = temperature_override
        elif self.config.get("temperature") is not None:
            payload["temperature"] = self.config.get("temperature")

        # Add any other hypothetical Windsurf-specific parameters.
        if query_data.get("output_format"): # E.g., "html", "css", "javascript", "explanation"
            payload["output_format"] = query_data.get("output_format")
        if query_data.get("framework_version"): # E.g., "React 18", "Vue 3"
            payload["framework_version"] = query_data.get("framework_version")

        self.logger.debug(f"Windsurf API request payload: {payload}")

        # Make the API call using APIManager.
        # 'windsurf' is the service name to be configured in APIManager.
        # 'generate' is a hypothetical endpoint for this service.
        response_data = await self.api_manager.make_request(
            service_name='windsurf', 
            endpoint='generate',
            method="POST",
            data=payload
        )

        # Handle errors from APIManager or the API.
        if response_data.get("error"):
            self.logger.error(f"API request to Windsurf failed: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data) # Include error details if available
            }

        # Try to parse the successful response from Windsurf.
        # This part is speculative, assuming common response patterns.
        try:
            extracted_content = response_data.get("response", response_data.get("generated_content"))
            if extracted_content is None: # Fallback to another common field if primary ones are missing
                 extracted_content = response_data.get("text")

            if extracted_content is None: # If no content found
                self.logger.error(f"Failed to extract meaningful content from Windsurf response. Response (first 300 chars): {str(response_data)[:300]}")
                return {"status": "error", "message": "Invalid or empty response structure from Windsurf API."}

            self.logger.info(f"Successfully received and parsed response from Windsurf for query: '{user_query[:100]}...'")
            return {
                "status": "success",
                "content": extracted_content, # The primary content from the AI
                "raw_response": response_data # Optionally include the full raw API response
            }
        except Exception as e: # Catch unexpected errors during parsing
            self.logger.error(f"Error parsing Windsurf response: {e}. Response data (first 500 chars): {str(response_data)[:500]}", exc_info=True)
            return {"status": "error", "message": f"Error parsing Windsurf response: {e}"}

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore

    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_WindsurfTest")
            self.service_configs = { # Needed for APIManager.get_auth_header, etc.
                "windsurf": {"api_key": "dummy_windsurf_key", "base_url": "https://api.windsurf.ai/v1"}
            }

        async def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
            # Simulate async behavior: await asyncio.sleep(0.01)
            if service_name == "windsurf" and endpoint == "generate":
                if "error" in data.get("prompt","").lower(): # Simple error simulation
                     return {"error": "Simulated API Error", "message": "The prompt contained 'error'", "status_code": 400}

                focus = data.get("focus", "general")
                response_text = (f"Simulated Windsurf AI response for focus '{focus}'. "
                                 f"Query was: '{data.get('prompt', '')[:100]}...'")
                if focus == "css-styling":
                    response_text += "\n```css\n/* Sample generated CSS */\nbody { font-family: 'Arial', sans-serif; }\n```"
                elif focus == "react":
                     response_text += "\n```jsx\n// Sample generated React component\nconst MyComponent = () => <div>Hello Windsurf!</div>;\nexport default MyComponent;\n```"
                
                return {
                    "id": "wind_xxxxxxxxxxxxxxxxx",
                    "response": response_text, # Main field
                    "generated_content": response_text, # Alternative field
                    "model_used": data.get("model", "windsurf-default-simulated"),
                    "focus_applied": focus,
                    "usage": {"prompt_tokens": len(data.get("prompt","")), "completion_tokens": len(response_text)}
                }
            return {"error": "Unknown service or endpoint in DummyAPIManager", "status_code": 404}

async def main_windsurf_test(): # Wrapped in async main function
    print("--- Testing WindsurfAgent ---")
    
    dummy_api_manager = DummyAPIManager()
    agent_config = {
        "model": "windsurf-expert-v2",
        "focus": "general-web-dev", # Default focus for this agent instance
        "max_tokens": 1800,
        "temperature": 0.4,
        "default_system_prompt": "You are a Windsurf AI, expert in all things web."
    }
    
    windsurf_agent = WindsurfAgent(
        agent_name="TestWindsurfAgent001",
        api_manager=dummy_api_manager, # type: ignore
        config=agent_config
    )

    print(f"Agent Name: {windsurf_agent.get_name()}")
    print(f"Agent Capabilities: {windsurf_agent.get_capabilities()}")
    print(f"Agent Configured Model: {windsurf_agent.model}, Focus: {windsurf_agent.focus_area}")

    # Test case 1: Web development query
    print("\n--- Test Case 1: Web Development Query ---")
    query1_data = {"prompt": "What are the best practices for responsive web design?"}
    response1 = await windsurf_agent.process_query(query1_data) # Awaited
    print(f"Response 1:\n{response1.get('content')}\n")
    assert response1["status"] == "success"
    assert "responsive web design" in response1.get("content", "") or "simulated Windsurf AI response" in response1.get("content", "")

    # Test case 2: Query with custom system prompt and focus override
    print("\n--- Test Case 2: Custom System Prompt & Focus Override ---")
    custom_sys_prompt = "You are a CSS animations guru."
    query2_data = {
        "prompt": "How can I create a smooth fade-in animation using CSS transitions?",
        "system_prompt": custom_sys_prompt,
        "focus": "css-styling" # Override agent's default focus
    }
    response2 = await windsurf_agent.process_query(query2_data) # Awaited
    print(f"Response 2:\n{response2.get('content')}\n")
    assert response2["status"] == "success"
    assert "css-styling" in response2.get("raw_response",{}).get("focus_applied","") # Check if focus was applied by dummy
    assert "fade-in animation" in response2.get("content","")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = await windsurf_agent.process_query(query3_data) # Awaited
    print(f"Response 3: {response3}\n")
    assert response3["status"] == "error"
    assert response3["message"] == "User query/prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error in Windsurf."} 
    response4 = await windsurf_agent.process_query(query4_data) # Awaited
    print(f"Response 4: {response4}\n")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- WindsurfAgent testing completed. ---")
    print("Note: This agent's implementation is based on a hypothetical Windsurf AI API.")
    print("API interaction details will likely need adjustments for a real API.")

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore
    # import asyncio # Added at the top

    if os.name == 'nt': # Optional: Windows specific policy for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_windsurf_test())
