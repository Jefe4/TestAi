# src/agents/windsurf_agent.py
"""Specialized agent for interacting with a hypothetical Windsurf AI, focusing on web development."""

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

class WindsurfAgent(BaseAgent):
    """
    An agent that utilizes a hypothetical Windsurf AI for tasks related to
    web development, UI/UX, frontend frameworks, and CSS.
    The Windsurf API details are speculative.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the WindsurfAgent.

        Args:
            agent_name: The name of the agent.
            api_manager: An instance of APIManager to handle API calls.
            config: Optional configuration dictionary for the agent.
                    Expected keys: "focus", "default_system_prompt", "model" (optional),
                                   "max_tokens", "temperature".
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager
        
        self.focus_area = self.config.get("focus", "web-development") # e.g., "web-development", "css-architecture", "react"
        self.model = self.config.get("model", "windsurf-latest") # Hypothetical model name
        self.default_system_prompt = self.config.get(
            "default_system_prompt", 
            "You are a web development expert. Provide solutions and best practices for frontend frameworks, CSS, and modern web practices."
        )
        self.logger.info(f"WindsurfAgent '{self.agent_name}' initialized with focus '{self.focus_area}' and model '{self.model}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the WindsurfAgent.
        """
        return {
            "description": "Agent specializing in web development, UI/UX, frontend frameworks, and CSS using a hypothetical Windsurf AI.",
            "capabilities": ["web_development", "ui_ux_design_principles", "frontend_framework_advice", "css_styling_techniques", "accessibility_best_practices"],
            "focus_areas_supported": ["general_web_dev", "react", "vue", "angular", "css_grid_flexbox", "web_performance"] # Example focus areas
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
        """
        Processes a query using the (hypothetical) Windsurf API.

        Args:
            query_data: A dictionary containing the query details.
                        Expected keys:
                        - "prompt" (str): The user's actual query.
                        - "system_prompt" (Optional[str]): Custom system prompt for this query.
                        - "focus" (Optional[str]): Override the agent's default focus area.
                        - Other potential keys: "project_context", "code_snippet", etc.

        Returns:
            A dictionary containing the status of the operation and the response content or error message.
        """
        user_query = query_data.get("prompt")
        if not user_query:
            self.logger.error("User query/prompt is missing in query_data.")
            return {"status": "error", "message": "User query/prompt missing"}

        system_prompt_override = query_data.get("system_prompt")
        system_prompt_to_use = system_prompt_override if system_prompt_override is not None else self.default_system_prompt
        
        current_focus = query_data.get("focus", self.focus_area)

        self.logger.info(f"Processing query for WindsurfAgent '{self.agent_name}' (focus: {current_focus}) with query: '{user_query[:100]}...'")

        # Similar to CursorAgent, assuming a combined prompt structure for now.
        # A real API might have more structured input.
        full_prompt = f"System Prompt:\n{system_prompt_to_use}\n\nUser Query:\n{user_query}"
        if query_data.get("context_details"): # Example of adding more context
            full_prompt += f"\n\nAdditional Context:\n{query_data.get('context_details')}"

        payload: Dict[str, Any] = {
            "prompt": full_prompt,
            "model": self.model, # If Windsurf API uses model selection
            "focus": current_focus, 
            "max_tokens": query_data.get("max_tokens", self.config.get("max_tokens", 2000)),
        }
        
        if self.config.get("temperature") is not None:
            payload["temperature"] = self.config.get("temperature")
        if query_data.get("temperature_override") is not None:
            payload["temperature"] = query_data.get("temperature_override")

        # Windsurf specific parameters (hypothetical)
        if query_data.get("output_format"): # e.g., "html", "css", "javascript", "explanation"
            payload["output_format"] = query_data.get("output_format")
        if query_data.get("framework_version"):
            payload["framework_version"] = query_data.get("framework_version")


        self.logger.debug(f"Windsurf API payload: {payload}")

        # Make the API call. Endpoint 'generate' or similar.
        # The service name 'windsurf' must be configured in APIManager.
        response_data = await self.api_manager.make_request( # Await the call
            service_name='windsurf', 
            endpoint='generate', # Hypothetical endpoint, e.g., /v1/generate
            method="POST",
            data=payload
        )

        if response_data.get("error"):
            self.logger.error(f"API request failed for Windsurf: {response_data.get('message', response_data.get('error'))}")
            return {
                "status": "error",
                "message": f"API request failed: {response_data.get('message', response_data.get('error'))}",
                "details": response_data.get("content", response_data)
            }

        try:
            # Assuming Windsurf API response might have a "response" or "generated_content" field.
            extracted_content = response_data.get("response", response_data.get("generated_content"))
            if extracted_content is None:
                 extracted_content = response_data.get("text") # Fallback

            if extracted_content is None:
                self.logger.error(f"Failed to extract content from Windsurf response. Response: {response_data}")
                return {"status": "error", "message": "Invalid response structure from Windsurf API."}

            self.logger.info(f"Successfully received and parsed response from Windsurf for query: '{user_query[:100]}...'")
            return {
                "status": "success",
                "content": extracted_content,
                "raw_response": response_data 
            }
        except Exception as e: 
            self.logger.error(f"Error parsing Windsurf response: {e}. Response data: {response_data}")
            return {"status": "error", "message": f"Error parsing Windsurf response: {e}"}

if __name__ == '__main__':
    from src.utils.logger import get_logger as setup_logger # type: ignore

    class DummyAPIManager:
        def __init__(self):
            self.logger = setup_logger("DummyAPIManager_WindsurfTest")
            self.service_configs = { # Needed for APIManager.get_auth_header, etc.
                "windsurf": {"api_key": "dummy_windsurf_key", "base_url": "https://api.windsurf.ai/v1"}
            }

        def make_request(self, service_name: str, endpoint: str, method: str, data: Dict[str, Any]) -> Dict[str, Any]:
            self.logger.info(f"DummyAPIManager received request for {service_name} -> {endpoint} with method {method}.")
            self.logger.debug(f"Request data: {data}")
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
    response1 = windsurf_agent.process_query(query1_data)
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
    response2 = windsurf_agent.process_query(query2_data)
    print(f"Response 2:\n{response2.get('content')}\n")
    assert response2["status"] == "success"
    assert "css-styling" in response2.get("raw_response",{}).get("focus_applied","") # Check if focus was applied by dummy
    assert "fade-in animation" in response2.get("content","")

    # Test case 3: Missing prompt
    print("\n--- Test Case 3: Missing Prompt ---")
    query3_data = {} # No prompt
    response3 = windsurf_agent.process_query(query3_data)
    print(f"Response 3: {response3}\n")
    assert response3["status"] == "error"
    assert response3["message"] == "User query/prompt missing"
    
    # Test case 4: API Error simulation
    print("\n--- Test Case 4: API Error ---")
    query4_data = {"prompt": "This prompt will cause an error in Windsurf."} 
    response4 = windsurf_agent.process_query(query4_data)
    print(f"Response 4: {response4}\n")
    assert response4["status"] == "error"
    assert "Simulated API Error" in response4.get("message", "")

    print("\n--- WindsurfAgent testing completed. ---")
    print("Note: This agent's implementation is based on a hypothetical Windsurf AI API.")
    print("API interaction details will likely need adjustments for a real API.")
