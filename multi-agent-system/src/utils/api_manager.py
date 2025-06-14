# src/utils/api_manager.py
"""
Manages API interactions for various external services.

This module provides the APIManager class, which is responsible for:
- Loading service configurations (API keys, base URLs) from YAML files and environment variables.
- Constructing appropriate authentication headers for different services.
- Making asynchronous HTTP requests using httpx.
- Handling common HTTP errors and timeouts.
- Basic rate limiting checks (placeholder for more advanced handling).
"""

import os
import json
import yaml
import httpx
import asyncio
from typing import Dict, Optional, Any, List # Standard typing modules, Added List

try:
    from ..utils.logger import get_logger # For structured logging
except ImportError: # Fallback for direct script execution or if logger structure is different
    import logging
    def get_logger(name): # Basic logger implementation for fallback
        logger = logging.getLogger(name)
        if not logger.handlers: # Ensure handler is added only once
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO) # Default to INFO level for fallback
        return logger

# Default path for the service configurations YAML file.
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "agent_configs.yaml")

class APIManager:
    """
    Handles API requests, authentication, and configuration for multiple external services.

    This class centralizes API interaction logic, making it easier for agents
    to communicate with different AI models and other web services without
    needing to manage HTTP clients, headers, or error handling themselves.
    It uses `httpx` for asynchronous HTTP requests.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the APIManager.

        Loads service configurations and sets up a common User-Agent.

        Args:
            config_path: Optional path to a YAML file containing service configurations
                         (e.g., API keys, base URLs). If None, attempts to load from
                         `DEFAULT_CONFIG_PATH`.
        """
        self.logger = get_logger("APIManager")
        self.service_configs: Dict[str, Dict[str, Any]] = {} # Stores configurations for each service
        self.user_agent = "MultiAgentSystem/1.0" # Standard User-Agent for requests
        
        actual_config_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        self.load_service_configs(actual_config_path)
        self.logger.info(f"APIManager initialized. Loaded configurations for services: {list(self.service_configs.keys())}")

    def load_service_configs(self, config_path: str):
        """
        Loads service configurations from a specified YAML file and environment variables.

        The loading priority is as follows:
        1. Configurations from the YAML file.
        2. Environment variables (can override file configurations if a service/key is not fully defined in the file).
        3. Hardcoded default base URLs (if not found in file or environment).

        API keys are particularly expected to be set either in the YAML file or as environment
        variables (e.g., `DEEPSEEK_API_KEY`).

        Args:
            config_path: The file path to the YAML configuration file.
        """
        self.logger.info(f"Attempting to load service configurations from: {config_path}")
        
        file_configs: Dict[str, Any] = {} # To store configs loaded from the YAML file
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_yaml = yaml.safe_load(f)
                    if loaded_yaml and isinstance(loaded_yaml, dict):
                        file_configs = loaded_yaml
                        self.logger.info(f"Successfully loaded configurations for {list(file_configs.keys())} from {config_path}")
                    else:
                        self.logger.warning(f"YAML file {config_path} is empty or not a dictionary. Skipping file load.")
            else:
                self.logger.info(f"Configuration file not found: {config_path}. Proceeding with environment variables and defaults.")
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML configuration file {config_path}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration file {config_path}: {e}")

        # Update service_configs with file_configs first, so environment/defaults can override if needed.
        self.service_configs.update(file_configs)

        # Define known services and their expected environment variables / default structures.
        # This provides a centralized place to manage service-specific details.
        services_to_configure = {
            "deepseek": {"api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1"},
            "cursor": {"api_key_env": "CURSOR_API_KEY", "base_url": "https://api.cursor.ai/v1"}, # Fictional URL
            "windsurf": {"api_key_env": "WINDSURF_API_KEY", "base_url": "https://api.windsurf.ai/v1"}, # Fictional URL
            "claude": {"api_key_env": "CLAUDE_API_KEY", "base_url": "https://api.anthropic.com/v1"},
            "gemini": {"api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
            # Add other services here
        }

        for service_name, details in services_to_configure.items():
            # If the service or its essential keys (api_key, base_url) are not fully defined
            # from the YAML file, try to load them from environment variables or use hardcoded defaults.
            if service_name not in self.service_configs or \
               self.service_configs[service_name].get("api_key") is None or \
               self.service_configs[service_name].get("base_url") is None:
                
                current_service_config = self.service_configs.get(service_name, {}) # Get existing partial config from file or empty dict
                
                # API key: existing file config OR environment variable OR placeholder
                api_key = current_service_config.get("api_key") or \
                            os.getenv(details["api_key_env"], f"YOUR_{service_name.upper()}_KEY_HERE")
                # Base URL: existing file config OR hardcoded default from `services_to_configure`
                base_url = current_service_config.get("base_url") or details["base_url"]
                
                # Update the service_configs entry, preserving any other settings from the file
                self.service_configs[service_name] = {
                    **current_service_config, # Start with existing settings from file
                    "api_key": api_key,       # Override/set api_key
                    "base_url": base_url      # Override/set base_url
                }
                
                if api_key == f"YOUR_{service_name.upper()}_KEY_HERE":
                    self.logger.warning(f"API key for '{service_name}' is a placeholder. "
                                        f"Set the environment variable '{details['api_key_env']}' or update the configuration file.")
                # Log if the service configuration was primarily established using env/defaults (not fully from file)
                elif service_name not in file_configs or \
                     not file_configs.get(service_name, {}).get("api_key") or \
                     not file_configs.get(service_name, {}).get("base_url"):
                    self.logger.info(f"Configuration for '{service_name}' established or completed using environment variables/defaults.")


    def get_auth_header(self, service_name: str) -> Dict[str, str]:
        """
        Constructs the appropriate authentication header for the specified service.

        Different services use different authentication schemes (e.g., Bearer token,
        custom header like x-api-key). This method centralizes that logic.

        Args:
            service_name: The name of the service (e.g., "claude", "deepseek").

        Returns:
            A dictionary representing the authentication header(s) for the service.
            Returns an empty dictionary if the service is not configured or API key is missing.
        """
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("api_key"):
            self.logger.warning(f"API key not found or is empty for service: '{service_name}'. Cannot generate auth header.")
            return {}

        api_key = service_info["api_key"]
        # Warn if a placeholder API key is being used.
        if api_key.startswith("YOUR_") and api_key.endswith("_KEY_HERE"):
             self.logger.warning(f"Using a placeholder API key for service '{service_name}' to generate auth header.")

        # Determine auth scheme based on service name
        if service_name in ["deepseek", "cursor", "gemini"]: # Services using Bearer token
            return {"Authorization": f"Bearer {api_key}"}
        elif service_name == "claude": # Claude uses x-api-key and a version header
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01"} # Example version
        elif service_name == "windsurf": # Hypothetical service with custom token
            return {"X-Custom-Auth-Token": api_key} 
        
        # If service name is unknown or auth scheme not defined
        self.logger.warning(f"Authentication header style not explicitly defined for service: '{service_name}'. Returning empty auth header.")
        return {}

    async def make_request(
        self,
        service_name: str,
        endpoint: str,
        method: str = "POST", # Default to POST as it's common for LLM APIs
        data: Optional[Dict[str, Any]] = None, # JSON payload for POST/PUT/PATCH
        params: Optional[Dict[str, Any]] = None, # URL query parameters
        extra_headers: Optional[Dict[str, str]] = None, # Additional headers to merge
        timeout: int = 30 # Request timeout in seconds
    ) -> Dict[str, Any]:
        """
        Makes an asynchronous HTTP request to the specified service endpoint using `httpx`.

        Handles constructing the full URL, adding authentication and standard headers,
        and processing the response, including error handling.

        Args:
            service_name: The name of the service as configured (e.g., "claude").
            endpoint: The API endpoint path (e.g., "messages", "chat/completions").
            method: The HTTP method (e.g., "GET", "POST").
            data: Optional dictionary for the JSON request body.
            params: Optional dictionary for URL query parameters.
            extra_headers: Optional dictionary of extra headers to include.
            timeout: Request timeout in seconds.

        Returns:
            A dictionary containing the JSON response from the API, or an error structure
            if the request fails. Example success: `{"status": "success", "data": ..., "status_code": 200}`.
            Example error: `{"error": "HTTPError", "message": ..., "status_code": ...}`.
        """
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("base_url"):
            self.logger.error(f"Base URL not configured for service: '{service_name}'. Cannot make request.")
            return {"error": f"Configuration missing or incomplete for {service_name}", "status_code": 500}

        base_url = service_info["base_url"]
        # Construct full URL, ensuring no double slashes between base_url and endpoint
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        auth_header = self.get_auth_header(service_name)
        # Standard headers: User-Agent, Content-Type (for JSON). Merge with auth and extra headers.
        client_headers = {"User-Agent": self.user_agent, "Content-Type": "application/json", **auth_header}
        if extra_headers: # Merge any additional headers provided by the caller
            client_headers.update(extra_headers)

        # Prepare keyword arguments for httpx.AsyncClient.request
        request_kwargs: Dict[str, Any] = {}
        if data is not None and method.upper() in ["POST", "PUT", "PATCH", "DELETE"]: # Common methods with JSON body
            request_kwargs["json"] = data
        if params is not None: # URL parameters for any method type
            request_kwargs["params"] = params
            
        self.logger.debug(f"Making async {method.upper()} request to {url}. "
                          f"Payload (first 100 chars if JSON): {str(request_kwargs.get('json'))[:100] if request_kwargs.get('json') else 'None'}, "
                          f"Params: {request_kwargs.get('params')}")

        try:
            # Use an async context manager for the httpx client
            async with httpx.AsyncClient(timeout=timeout, headers=client_headers) as client:
                response = await client.request(method=method.upper(), url=url, **request_kwargs)

                # Placeholder for rate limiting checks based on response headers
                self.handle_rate_limiting(service_name, response.headers)

                response.raise_for_status() # Raises `httpx.HTTPStatusError` for 4xx/5xx responses
            
                if not response.content: # Handle successful but empty responses
                     self.logger.info(f"Received empty but successful (status {response.status_code}) response from '{service_name}' for endpoint '{endpoint}'.")
                     # Return a standardized success format for empty content
                     return {"status": "success", "data": None, "status_code": response.status_code}

                # Attempt to parse JSON response
                # Note: response.json() is called here. If it fails, the JSONDecodeError below is caught.
                return response.json()

        # Specific error handling for HTTP status errors (4xx, 5xx)
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP status error for '{service_name}' on {url}: {e.response.status_code} - Response (first 500 chars): {e.response.text[:500]}")
            error_content: Any = e.response.text # Default to raw text if JSON parsing fails
            try:
                error_content = e.response.json() # Try to get structured JSON error from response
            except json.JSONDecodeError:
                self.logger.warning(f"Could not parse JSON from error response for '{service_name}' on {url}. Raw text used.")
            return {"error": "HTTPStatusError", "message": str(e), "status_code": e.response.status_code, "content": error_content}

        # Handle request timeouts
        except httpx.TimeoutException as e:
            self.logger.error(f"Timeout error for '{service_name}' on {url} after {timeout} seconds: {e}")
            return {"error": "TimeoutException", "message": f"Request timed out after {timeout} seconds.", "status_code": 408} # HTTP 408 Request Timeout

        # Handle other httpx request-related errors (network issues, DNS failures, etc.)
        except httpx.RequestError as e:
            self.logger.error(f"Request error for '{service_name}' on {url}: {e}")
            return {"error": "RequestError", "message": str(e), "status_code": 503} # HTTP 503 Service Unavailable (generic for request issues)

        # Handle errors if the successful response content is not valid JSON
        except json.JSONDecodeError as e:
            # This error occurs if response.json() fails inside the try block after a successful HTTP status.
            # 'response' object should be available here from the try block.
            raw_response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "N/A"
            self.logger.error(f"Failed to decode JSON response from '{service_name}' for '{url}': {e}. Response text (first 500 chars): {raw_response_text[:500]}")
            return {"error": "JSONDecodeError", "message": str(e), "raw_response": raw_response_text, "status_code": 502} # HTTP 502 Bad Gateway (malformed upstream response)

    def handle_rate_limiting(self, service_name: str, response_headers: httpx.Headers):
        """
        Basic placeholder for handling rate limiting based on response headers.

        This method checks for common rate limit headers like 'X-RateLimit-Remaining'
        and 'Retry-After'. A more robust implementation would involve strategies
        like pausing requests, queuing, or exponential backoff.

        Args:
            service_name: The name of the service for which the request was made.
            response_headers: The `httpx.Headers` object from the response.
        """
        remaining = response_headers.get('X-RateLimit-Remaining') # How many requests are left in the current window
        retry_after_seconds = response_headers.get('Retry-After') # Seconds to wait or a specific date/time

        if remaining is not None:
            try:
                if int(remaining) == 0:
                    self.logger.warning(f"Rate limit potentially hit for service '{service_name}' (X-RateLimit-Remaining is 0). "
                                        f"Consider pausing. Retry-After: {retry_after_seconds}")
            except ValueError:
                self.logger.warning(f"Could not parse X-RateLimit-Remaining value: '{remaining}' for service '{service_name}'.")
        elif retry_after_seconds is not None:
             self.logger.warning(f"Rate limit potentially hit for service '{service_name}'. "
                                 f"'Retry-After' header found: {retry_after_seconds}. Consider pausing requests.")
        # A full implementation would involve coordination to pause future requests to this service.
        # For now, this method only logs warnings.

    def _convert_messages_to_gemini_contents(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Converts a list of messages (common format: {"role": ..., "content": ...})
        to Gemini's "contents" format ({"role": ..., "parts": [{"text": ...}]}).

        System messages are ignored here as they are handled by `systemInstruction` in Gemini API.
        Unknown roles default to "user".
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role.lower() == "system":
                self.logger.debug("System message in `messages` list ignored during Gemini contents conversion. Use 'system_prompt' kwarg for Gemini's systemInstruction.")
                continue # Skip system messages for the 'contents' array

            # Adapt role if needed, Gemini typically uses 'user' and 'model'
            # The Gemini API requires alternating user and model roles.
            # This basic conversion assumes the input `messages` list might not strictly adhere to this
            # for all models, but for a direct call to Gemini, the caller should ideally ensure this.
            # For simplicity, we map 'assistant' or other non-user roles to 'model'.
            # A more robust solution might validate/enforce alternating roles if strictly for Gemini.
            gemini_role = "user" if role.lower() == "user" else "model"

            contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        return contents

    async def call_llm_service(
        self,
        service_name: str,
        model_name: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Provides a simplified interface for making chat/completion calls to common LLM services.

        This method abstracts the service-specific payload construction.

        Args:
            service_name: The name of the LLM service (e.g., "claude", "gemini", "deepseek").
            model_name: The specific model to use for the service.
            messages: A list of message dictionaries (e.g., [{"role": "user", "content": "Hello"}]).
                      The structure should be adaptable to the target service's requirements.
                      For Gemini, this will be converted to its 'contents' format.
            max_tokens: The maximum number of tokens to generate in the response.
            temperature: The sampling temperature for generation (0.0 to 1.0+).
            **kwargs: Additional keyword arguments specific to the LLM service.
                      For "claude", can include "system_prompt".
                      For "gemini", can include "system_prompt", "top_p", "top_k".

        Returns:
            A dictionary containing the response from the LLM service, typically including
            status, content, and any error information.
        """
        self.logger.info(f"call_llm_service invoked for service: {service_name}, model: {model_name}")

        endpoint = ""
        payload = {}
        method = "POST" # Common for chat/completion LLMs

        if service_name == "claude":
            endpoint = "messages"
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if "system_prompt" in kwargs:
                payload["system"] = kwargs.pop("system_prompt")
            payload.update(kwargs) # Add any other remaining kwargs

        elif service_name == "gemini":
            # Note: This uses the REST API for Gemini, not the Python SDK directly here.
            # APIManager is generally for HTTP REST calls.
            # Gemini SDK calls are usually made directly from the GeminiAgent.
            endpoint = f"models/{model_name}:generateContent" # Check Gemini REST API docs for exact endpoint

            gemini_contents = self._convert_messages_to_gemini_contents(messages)
            if not gemini_contents and any(msg.get("role", "").lower() != "system" for msg in messages):
                self.logger.error(f"For Gemini, 'messages' resulted in empty 'contents' after filtering system prompts. Original messages: {messages}")
                return {"status": "error", "message": "Gemini 'contents' cannot be empty. Provide user/model messages."}

            payload = {
                "contents": gemini_contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": kwargs.pop("top_p", 1.0),
                    "topK": kwargs.pop("top_k", None), # top_k often optional
                },
            }
            if kwargs.get("system_prompt"):
                payload["systemInstruction"] = {"parts": [{"text": kwargs.pop("system_prompt")}]}

            # Remove None values from generationConfig as Gemini API might not like nulls for optional fields
            payload["generationConfig"] = {k: v for k, v in payload["generationConfig"].items() if v is not None}
            payload.update(kwargs)


        elif service_name == "deepseek":
            endpoint = "chat/completions"
            payload = {
                "model": model_name,
                "messages": messages, # DeepSeek uses OpenAI-like message format
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            payload.update(kwargs)

        else:
            self.logger.error(f"LLM service '{service_name}' is not recognized by call_llm_service.")
            return {"status": "error", "message": f"Service '{service_name}' not supported by call_llm_service."}

        if not endpoint: # Should be caught by the else above, but as a safeguard
            self.logger.error(f"Endpoint not defined for LLM service '{service_name}'.")
            return {"status": "error", "message": f"Endpoint configuration missing for service '{service_name}'."}

        return await self.make_request(
            service_name=service_name,
            endpoint=endpoint,
            method=method,
            data=payload
        )


if __name__ == '__main__':
    # This is for basic demonstration and testing of the APIManager
    # Ensure that a dummy `agent_configs.yaml` can be created in `src/config/`
    # or provide a valid path to an existing one.
    # Set relevant environment variables (e.g., DEEPSEEK_API_KEY) to test actual API calls.

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")
    dummy_config_dir = os.path.join(project_root, "src", "config")
    dummy_config_path = os.path.join(dummy_config_dir, "agent_configs.yaml")

    print(f"Looking for dummy config at: {dummy_config_path}")

    if not os.path.exists(dummy_config_dir):
        try:
            os.makedirs(dummy_config_dir)
            print(f"Created directory: {dummy_config_dir}")
        except OSError as e:
            print(f"Error creating directory {dummy_config_dir}: {e}")
            # Potentially exit or handle error if directory creation is critical for the test

    # Create a sample config if it doesn't exist, for testing purposes
    if not os.path.exists(dummy_config_path):
        sample_config_data = {
            "deepseek": {
                "api_key": "YOUR_DEEPSEEK_KEY_FROM_FILE_OR_ENV_IN_MAIN",
                "base_url": "https://api.deepseek.com/v1"
            },
            "claude": {
                "api_key": os.getenv("CLAUDE_API_KEY", "YOUR_CLAUDE_KEY_FROM_FILE_OR_ENV_IN_MAIN")
            },
            # Example for a service that might only use env vars / defaults
            "gemini": {}
        }
        try:
            with open(dummy_config_path, 'w') as f:
                yaml.dump(sample_config_data, f)
            print(f"Created dummy config file at {dummy_config_path} for testing.")
        except IOError as e:
            print(f"Error writing dummy config file {dummy_config_path}: {e}")
    else:
        print(f"Dummy config file already exists at {dummy_config_path}.")

    # Example: Set dummy env vars for services not fully defined in file, for testing load priority
    os.environ["CURSOR_API_KEY"] = "env_cursor_key_123_for_test" # This should be picked up
    # Ensure GEMINI_API_KEY is set in your environment if you want to test Gemini loading from env
    # os.environ["GEMINI_API_KEY"] = "env_gemini_key_xyz_for_test"

    manager = APIManager(config_path=dummy_config_path)

    print("\n--- Service Configurations Loaded by APIManager ---")
    for service, conf in manager.service_configs.items():
        # Mask API keys for printing
        masked_key = str(conf.get('api_key', 'N/A'))
        if len(masked_key) > 8 and not masked_key.startswith("YOUR_"): # Basic masking
            masked_key = masked_key[:4] + '...' + masked_key[-4:]
        elif masked_key.startswith("YOUR_"):
            masked_key = "[PLACEHOLDER]"

        print(f"  Service: {service}")
        print(f"    Base URL: {conf.get('base_url', 'N/A')}")
        print(f"    API Key (masked): {masked_key}")
        # Print other custom configs if any
        other_configs = {k:v for k,v in conf.items() if k not in ['api_key', 'base_url']}
        if other_configs: print(f"    Other configs: {other_configs}")


    print("\n--- Auth Headers (Examples) ---")
    # Test auth header generation for a few services
    for service_test_name in ["deepseek", "claude", "gemini", "windsurf", "cursor", "unknown_service"]:
        print(f"  Auth header for '{service_test_name}': {manager.get_auth_header(service_test_name)}")

    print("\n--- Testing make_request (with a public JSONPlaceholder API) ---")
    # Add a temporary public API config for testing make_request directly without needing real keys
    manager.service_configs["public_jsonplaceholder"] = {
        "base_url": "https://jsonplaceholder.typicode.com",
        "api_key": "unused_dummy_key_for_public_api" # Not actually used by JSONPlaceholder
    }

    async def main_test():
        print("  Testing GET request to JSONPlaceholder...")
        get_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="todos/1", # Example GET endpoint
            method="GET"
        )
        print(f"  GET Response: {json.dumps(get_response, indent=2)}")

        print("\n  Testing POST request to JSONPlaceholder...")
        post_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="posts", # Example POST endpoint
            method="POST",
            data={"title": "Test Post by APIManager", "body": "This is a test.", "userId": 1}
        )
        print(f"  POST Response: {json.dumps(post_response, indent=2)}")

        print("\n  Testing GET request with params to JSONPlaceholder...")
        get_params_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="comments", # Get comments, filter by postId
            method="GET",
            params={"postId": 1}
        )
        print(f"  GET with Params Response (first item if many): {json.dumps(get_params_response[0] if isinstance(get_params_response, list) and get_params_response else get_params_response, indent=2)}")


        print("\n  Testing non-existent endpoint (should result in 404)...")
        not_found_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="nonexistent/endpoint/123",
            method="GET"
        )
        print(f"  404 Not Found Response: {json.dumps(not_found_response, indent=2)}")

        # Example of testing a timeout (requires a slow endpoint or very short timeout)
        # manager.service_configs["httpbin_delay"] = {"base_url": "https://httpbin.org", "api_key": "dummy"}
        # print("\n  Testing Timeout (will take a few seconds)...")
        # timeout_response = await manager.make_request(
        #     service_name="httpbin_delay",
        #     endpoint="delay/5", # httpbin endpoint that delays response by 5 seconds
        #     method="GET",
        #     timeout=2 # Set timeout to 2 seconds to trigger TimeoutException
        # )
        # print(f"  Timeout Response: {json.dumps(timeout_response, indent=2)}")


    if __name__ == '__main__': # Ensure this check is here to run the async main_test
        # Standard asyncio setup
        if os.name == 'nt': # Optional: Windows specific policy for asyncio if needed
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Add tests for call_llm_service
        async def run_all_tests():
            await main_test() # Run existing tests

            print("\n--- Testing call_llm_service ---")
            messages_example = [{"role": "user", "content": "Hello, what is your name?"}]

            # Mock the make_request for these tests if not using real keys or if DummyAPIManager needs extension
            original_make_request = manager.make_request

            async def mock_llm_make_request(service_name, endpoint, method, data, **kwargs):
                print(f"Mocked make_request for call_llm_service: {service_name}, {endpoint}, DATA: {data}")
                if service_name == "claude" and endpoint == "messages":
                    return {"status": "success", "content": [{"type": "text", "text": f"Mock Claude response to: {data['messages'][0]['content']}"}]}
                if service_name == "gemini" and endpoint.startswith("models/") and endpoint.endswith(":generateContent"):
                     # Gemini's actual response is more complex, this is simplified
                    return {"status": "success", "candidates": [{"content": {"parts": [{"text": f"Mock Gemini response to: {data['contents'][0]['parts'][0]['text']}"}]}}]}
                if service_name == "deepseek" and endpoint == "chat/completions":
                    return {"status": "success", "choices": [{"message": {"content": f"Mock DeepSeek response to: {data['messages'][0]['content']}"}}]}
                return {"status": "error", "message": f"Service {service_name} endpoint {endpoint} not mocked for call_llm_service test"}

            manager.make_request = mock_llm_make_request

            if "claude" not in manager.service_configs: manager.service_configs["claude"] = {"api_key":"dummy", "base_url":"dummy"}
            llm_response_claude = await manager.call_llm_service(
                service_name="claude", model_name="claude-test-model",
                messages=messages_example, max_tokens=50, system_prompt="You are a helpful assistant."
            )
            print(f"  LLM Service Response (Claude via mock): {llm_response_claude}")

            if "gemini" not in manager.service_configs: manager.service_configs["gemini"] = {"api_key":"dummy", "base_url":"dummy"}
            gemini_messages_example = [{"role": "user", "content": "Write a short poem about AI."}]
            llm_response_gemini = await manager.call_llm_service(
                service_name="gemini", model_name="gemini-test-model",
                messages=gemini_messages_example, max_tokens=60, system_prompt="You are a poetic AI."
            )
            print(f"  LLM Service Response (Gemini via mock): {llm_response_gemini}")

            # Test Gemini with a system message in the list (should be ignored by _convert_messages_to_gemini_contents)
            gemini_messages_with_system = [
                {"role": "system", "content": "This should be ignored."},
                {"role": "user", "content": "Tell me a joke."}
            ]
            llm_response_gemini_sys = await manager.call_llm_service(
                service_name="gemini", model_name="gemini-test-model",
                messages=gemini_messages_with_system, max_tokens=60
            )
            print(f"  LLM Service Response (Gemini with system message in list via mock): {llm_response_gemini_sys}")


            if "deepseek" not in manager.service_configs: manager.service_configs["deepseek"] = {"api_key":"dummy", "base_url":"dummy"}
            llm_response_deepseek = await manager.call_llm_service(
                service_name="deepseek", model_name="deepseek-test-model",
                messages=messages_example, max_tokens=50
            )
            print(f"  LLM Service Response (DeepSeek via mock): {llm_response_deepseek}")

            # Test unsupported service
            llm_response_unknown = await manager.call_llm_service(
                service_name="unknown_llm", model_name="unknown_model",
                messages=messages_example, max_tokens=50
            )
            print(f"  LLM Service Response (Unknown via mock): {llm_response_unknown}")

            manager.make_request = original_make_request # Restore original make_request

        asyncio.run(run_all_tests()) # Run all tests including new ones

        print("\n--- APIManager basic testing done. ---")
    # These prints are outside the if __name__ == '__main__' guard, which is unusual.
    # They will run if the file is imported, which might not be intended.
    # Consider moving them inside the guard or removing if they are only for direct execution.
    # print(f"Note: A dummy config may have been created/used at {dummy_config_path}. You can remove it if desired.")
    # print("To test specific service configurations, ensure their API keys are set as environment variables (e.g., DEEPSEEK_API_KEY, CLAUDE_API_KEY, etc.) or in the YAML config.")

# TODOs from original plan (some might be addressed by recent changes):
# - Consider more sophisticated API key management (e.g., dedicated secrets manager).
# - Implement more robust retry mechanisms (e.g., exponential backoff using a library like 'tenacity').
# - Expand handle_rate_limiting to actually pause/retry requests based on headers.
# - Securely handle and log API errors, potentially redacting sensitive parts of requests/responses in logs.
# - The use of httpx.AsyncClient per request is simple but less efficient than a shared client instance.
#   For high-throughput scenarios, consider managing a shared AsyncClient lifecycle,
#   though this adds complexity (e.g., ensuring it's closed properly).
#   For this system's current design, per-request client is acceptable.
        retry_after_seconds = response_headers.get('Retry-After') # Can be seconds or a date

        if remaining is not None:
            try:
                if int(remaining) == 0:
                    self.logger.warning(f"Rate limit potentially hit for {service_name} (remaining is 0). Retry-After: {retry_after_seconds}")
            except ValueError:
                self.logger.warning(f"Could not parse X-RateLimit-Remaining value: {remaining} for {service_name}")
        elif retry_after_seconds is not None:
             self.logger.warning(f"Rate limit potentially hit for {service_name}. Retry-After header found: {retry_after_seconds}")
        # Actual implementation would involve pausing requests or queuing.

if __name__ == '__main__':
    # This is for basic testing of the APIManager
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..") # Assuming src/utils -> multi-agent-system
    dummy_config_dir = os.path.join(project_root, "src", "config")
    dummy_config_path = os.path.join(dummy_config_dir, "agent_configs.yaml")

    print(f"Looking for dummy config at: {dummy_config_path}")

    if not os.path.exists(dummy_config_dir):
        os.makedirs(dummy_config_dir)
        print(f"Created directory: {dummy_config_dir}")

    if not os.path.exists(dummy_config_path):
        sample_config_data = {
            "deepseek": {
                "api_key": "YOUR_DEEPSEEK_KEY_FROM_FILE", # Will be overridden by env var if DEEPSEEK_API_KEY is set
                "base_url": "https://api.deepseek.com/v1/file_config_test" 
            },
            "claude": {
                "api_key": os.getenv("CLAUDE_API_KEY", "YOUR_CLAUDE_KEY_FROM_FILE_OR_ENV_IN_MAIN")
                # base_url will use hardcoded default if not here and not in env
            },
            "gemini": {
                # api_key will use env or placeholder
                # base_url will use hardcoded default
            }
        }
        with open(dummy_config_path, 'w') as f:
            yaml.dump(sample_config_data, f)
        print(f"Created dummy config file at {dummy_config_path} for testing.")
    else:
        print(f"Dummy config file already exists at {dummy_config_path}.")

    # Set dummy env vars for services not fully defined in file, for testing load priority
    os.environ["CURSOR_API_KEY"] = "env_cursor_key_123"
    os.environ["GEMINI_API_KEY"] = "env_gemini_key_xyz" # This should be used if not in file

    manager = APIManager(config_path=dummy_config_path) 
    
    print("\n--- Service Configurations Loaded ---")
    for service, conf in manager.service_configs.items():
        masked_key = str(conf.get('api_key', ''))[:5] + '...' if conf.get('api_key') else 'N/A'
        print(f"  Service: {service}")
        print(f"    Base URL: {conf.get('base_url', 'N/A')}")
        print(f"    API Key (masked): {masked_key}")
        if 'other_config' in conf: print(f"    Other: {conf['other_config']}")


    print("\n--- Auth Headers (Examples) ---")
    for service in ["deepseek", "claude", "gemini", "windsurf", "cursor", "unknown_service"]:
        print(f"  {service}: {manager.get_auth_header(service)}")

    print("\n--- Testing make_request (with public API) ---")
    # Add a temporary public API config for testing make_request directly
    manager.service_configs["public_jsonplaceholder"] = {
        "base_url": "https://jsonplaceholder.typicode.com",
        "api_key": "unused_dummy_key" 
    }

    async def main_test(): # Create an async function to run the tests
        print("  Testing GET request...")
        get_response = await manager.make_request( # await the async call
            service_name="public_jsonplaceholder",
            endpoint="todos/1",
            method="GET"
        )
        print(f"  GET Response: {json.dumps(get_response, indent=2)}")

        print("\n  Testing POST request...")
        post_response = await manager.make_request( # await the async call
            service_name="public_jsonplaceholder",
            endpoint="posts",
            method="POST",
            data={"title": "Test Post", "body": "This is a test.", "userId": 1}
        )
        print(f"  POST Response: {json.dumps(post_response, indent=2)}")

        print("\n  Testing GET request with params...")
        get_params_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="comments",
            method="GET",
            params={"postId": 1}
        )
        print(f"  GET with Params Response: {json.dumps(get_params_response, indent=2)}")


        print("\n  Testing non-existent endpoint (404)...")
        not_found_response = await manager.make_request(
            service_name="public_jsonplaceholder",
            endpoint="nonexistent/123",
            method="GET"
        )
        print(f"  404 Response: {json.dumps(not_found_response, indent=2)}")

        # Test timeout (requires a slow endpoint or very short timeout)
        # manager.service_configs["httpbin"] = {"base_url": "https://httpbin.org", "api_key": "dummy"}
        # print("\n  Testing Timeout...")
        # timeout_response = await manager.make_request(
        #     service_name="httpbin",
        #     endpoint="delay/5", # 5 second delay
        #     method="GET",
        #     timeout=2 # 2 second timeout
        # )
        # print(f"  Timeout Response: {json.dumps(timeout_response, indent=2)}")


    if __name__ == '__main__':
        # ... (rest of the setup code remains the same)
        # ...
        # Run the async main_test function
        asyncio.run(main_test())
        print("\n--- APIManager testing done. ---")
    print(f"Note: A dummy config may have been created/used at {dummy_config_path}. You can remove it if desired.")
    print("To test specific service configurations, ensure their API keys are set as environment variables (e.g., DEEPSEEK_API_KEY, CLAUDE_API_KEY, etc.) or in the YAML config.")

# TODOs from original plan:
# - Consider more sophisticated API key management.
# - Implement more robust retry mechanisms (e.g., exponential backoff).
# - Expand handle_rate_limiting to actually pause/retry.
# - Securely handle and log API errors.
# - Ensure thread-safety if used in a multi-threaded context (requests.Session is not thread-safe). -> httpx.AsyncClient is designed for asyncio
#   For asyncio, aiohttp.ClientSession would be preferred. -> Using httpx now
