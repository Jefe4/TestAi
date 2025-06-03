# src/utils/api_manager.py
"""Manages API interactions for various external services using asynchronous requests."""

import os
import json
import yaml
import asyncio # Added
import aiohttp # Added
from typing import Dict, Optional, Any

try:
    from ..utils.logger import get_logger
except ImportError: 
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "agent_configs.yaml")

class APIManager:
    """
    Handles API requests, authentication, and configuration for multiple services asynchronously.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("APIManager")
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        self.client_session: Optional[aiohttp.ClientSession] = None # Changed from requests.Session
        
        actual_config_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        self.load_service_configs(actual_config_path) # This remains synchronous
        self.logger.info(f"APIManager initialized. Loaded configs for: {list(self.service_configs.keys())}")

    async def _get_session(self) -> aiohttp.ClientSession: # New async method
        """Ensures an active aiohttp.ClientSession is available."""
        if self.client_session is None or self.client_session.closed:
            self.logger.info("Creating new aiohttp.ClientSession.")
            self.client_session = aiohttp.ClientSession(headers={"User-Agent": "MultiAgentSystem/1.0"})
        return self.client_session

    def load_service_configs(self, config_path: str): # Synchronous, as it's called in __init__
        self.logger.info(f"Attempting to load service configurations from: {config_path}")
        file_configs: Dict[str, Any] = {}
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

        self.service_configs.update(file_configs)
        services_to_configure = {
            "deepseek": {"api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1"},
            "cursor": {"api_key_env": "CURSOR_API_KEY", "base_url": "https://api.cursor.ai/v1"},
            "windsurf": {"api_key_env": "WINDSURF_API_KEY", "base_url": "https://api.windsurf.ai/v1"},
            "claude": {"api_key_env": "CLAUDE_API_KEY", "base_url": "https://api.anthropic.com/v1"},
            "gemini": {"api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
        }
        for service_name, details in services_to_configure.items():
            if service_name not in self.service_configs or \
               "api_key" not in self.service_configs[service_name] or \
               "base_url" not in self.service_configs[service_name]:
                current_service_config = self.service_configs.get(service_name, {})
                api_key = current_service_config.get("api_key") or \
                            os.getenv(details["api_key_env"], f"YOUR_{service_name.upper()}_KEY_HERE")
                base_url = current_service_config.get("base_url") or details["base_url"]
                self.service_configs[service_name] = {"api_key": api_key, "base_url": base_url, **current_service_config}
                if api_key == f"YOUR_{service_name.upper()}_KEY_HERE":
                    self.logger.warning(f"API key for {service_name} is placeholder. Set {details['api_key_env']} or update config.")
                elif service_name not in file_configs:
                    self.logger.info(f"Config for {service_name} from env/defaults.")

    def get_auth_header(self, service_name: str) -> Dict[str, str]: # Remains synchronous
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("api_key"):
            self.logger.warning(f"API key not found for service: {service_name}")
            return {}
        api_key = service_info["api_key"]
        if api_key.startswith("YOUR_") and api_key.endswith("_KEY_HERE"):
             self.logger.warning(f"Using placeholder API key for {service_name}.")
        if service_name in ["deepseek", "cursor", "gemini"]:
            return {"Authorization": f"Bearer {api_key}"}
        elif service_name == "claude":
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        elif service_name == "windsurf":
            return {"X-Custom-Auth-Token": api_key} 
        self.logger.warning(f"Auth header style not defined for: {service_name}.")
        return {}

    async def make_request( # Changed to async
        self, service_name: str, endpoint: str, method: str = "POST",
        data: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None, timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("base_url"):
            self.logger.error(f"Base URL not configured for service: {service_name}")
            return {"status": "error", "message": f"Configuration missing for {service_name}", "status_code": 500}

        base_url = service_info["base_url"]
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        auth_header = self.get_auth_header(service_name) # This is sync
        # Default content type for JSON, can be overridden by extra_headers or specific auth_header logic (like Claude's)
        final_headers = {"Content-Type": "application/json", **auth_header, **(extra_headers or {})}

        session = await self._get_session()
        
        req_kwargs: Dict[str, Any] = {}
        if method.upper() in ["POST", "PUT", "PATCH"] and data is not None:
            req_kwargs["json"] = data # aiohttp uses 'json' kwarg for dict payload
        # For GET/DELETE, aiohttp uses 'params' for query string. If 'data' is provided for GET/DELETE, assume it's for params.
        elif method.upper() in ["GET", "DELETE"] and data is not None:
            req_kwargs["params"] = data 
        
        if params is not None: # Explicit 'params' override or add to existing
            req_kwargs["params"] = {**(req_kwargs.get("params", {})), **params}
            
        self.logger.debug(f"Async API Request: {method.upper()} {url} | Headers: {final_headers} | JSON: {req_kwargs.get('json')} | Params: {req_kwargs.get('params')}")

        try:
            async with session.request(
                method.upper(), url, headers=final_headers, 
                timeout=aiohttp.ClientTimeout(total=timeout_seconds), **req_kwargs
            ) as response:
                self.logger.debug(f"API Response: {method.upper()} {url} | Status: {response.status} {response.reason}")
                self.handle_rate_limiting(service_name, response.headers) # Sync call, consider making async if it involves I/O

                response_content: Any
                content_type = response.headers.get('Content-Type', '').lower()
                if 'application/json' in content_type:
                    try:
                        response_content = await response.json()
                    except aiohttp.ContentTypeError: # If server sends wrong Content-Type for JSON
                        self.logger.warning(f"JSON decode error due to Content-Type: {content_type}. Reading as text.")
                        response_content = await response.text()
                else:
                    response_content = await response.text()

                if not (200 <= response.status < 300):
                    self.logger.error(f"API Error ({service_name}): {response.status} {response.reason} - {str(response_content)[:200]}")
                    return {"status": "error", "message": f"API request failed with status {response.status}: {str(response_content)}", "status_code": response.status}
                
                # If response_content is already a dict (from response.json()), return it directly.
                # Otherwise, wrap text content in a standard structure.
                if isinstance(response_content, dict):
                    # Ensure a "status" key if it's a successful dict response but doesn't conform to our agent responses
                    if "status" not in response_content : response_content["status"] = "success"
                    return response_content 
                else: # Text, XML, etc.
                    return {"status": "success", "content": response_content, "status_code": response.status}

        except aiohttp.ClientConnectorError as e: # More specific network errors
            self.logger.error(f"AIOHTTP ClientConnectorError ({service_name}) for {url}: {e}")
            return {"status": "error", "message": f"AIOHTTP ClientConnectorError: {str(e)}", "status_code": 503} # Service Unavailable
        except aiohttp.ClientError as e: # Generic aiohttp client error
            self.logger.error(f"AIOHTTP ClientError ({service_name}) for {url}: {e}")
            return {"status": "error", "message": f"AIOHTTP ClientError: {str(e)}", "status_code": 500} # Internal Server Error for general client issues
        except asyncio.TimeoutError:
            self.logger.error(f"Request to {service_name} ({url}) timed out after {timeout_seconds}s.")
            return {"status": "error", "message": "Request timed out", "status_code": 408} # Request Timeout
        except Exception as e:
            self.logger.error(f"Unexpected error during API request ({service_name}) for {url}: {e}", exc_info=True)
            return {"status": "error", "message": f"Unexpected error: {str(e)}", "status_code": 500}


    def handle_rate_limiting(self, service_name: str, response_headers: Any): 
        # This is CaseInsensitiveMultiDict from aiohttp/multidict
        remaining = response_headers.get('X-RateLimit-Remaining')
        retry_after = response_headers.get('Retry-After') 
        if remaining is not None:
            try:
                if int(remaining) == 0:
                    self.logger.warning(f"Rate limit potentially hit for {service_name} (remaining 0). Retry-After: {retry_after}")
            except ValueError:
                self.logger.warning(f"Could not parse X-RateLimit-Remaining: {remaining} for {service_name}")
        elif retry_after is not None:
             self.logger.warning(f"Rate limit potentially hit for {service_name}. Retry-After: {retry_after}")

    async def close_session(self) -> None: # New async method
        """Closes the aiohttp.ClientSession if it exists and is open."""
        if self.client_session and not self.client_session.closed:
            await self.client_session.close()
            self.logger.info("AIOHTTP client session closed.")
            self.client_session = None # Set to None after closing
        elif self.client_session and self.client_session.closed:
             self.logger.info("AIOHTTP client session was already closed.")
        else:
            self.logger.info("No active AIOHTTP client session to close.")


if __name__ == '__main__':
    # Updated __main__ to run async tests
    async def run_tests():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, "..", "..") 
        dummy_config_dir = os.path.join(project_root, "src", "config")
        dummy_config_path = os.path.join(dummy_config_dir, "agent_configs.yaml")

        # ... (config file creation logic from previous __main__ can be kept if needed) ...
        if not os.path.exists(dummy_config_path):
             print(f"Dummy config {dummy_config_path} not found. Create it for full __main__ test.")

        manager = APIManager(config_path=dummy_config_path)
        
        print("\n--- Service Configurations Loaded ---")
        # ... (print service configs) ...

        print("\n--- Auth Headers (Examples) ---")
        # ... (print auth headers) ...

        print("\n--- Testing make_request (with public API) ---")
        manager.service_configs["public_jsonplaceholder"] = {
            "base_url": "https://jsonplaceholder.typicode.com", "api_key": "unused"
        }
        
        print("  Testing GET request...")
        get_response = await manager.make_request( # await here
            service_name="public_jsonplaceholder", endpoint="todos/1", method="GET"
        )
        print(f"  GET Response: {json.dumps(get_response, indent=2)}")

        print("\n  Testing POST request...")
        post_response = await manager.make_request( # await here
            service_name="public_jsonplaceholder", endpoint="posts", method="POST",
            data={"title": "Test Post", "body": "This is a test.", "userId": 1}
        )
        print(f"  POST Response: {json.dumps(post_response, indent=2)}")

        await manager.close_session() # Important to close session

    if os.name == 'nt': # Windows patch for asyncio if needed
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_tests())
    print("\n--- APIManager async testing done. ---")
