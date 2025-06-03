# src/utils/api_manager.py
"""Manages API interactions for various external services."""

import os
import json
import yaml
import requests
from typing import Dict, Optional, Any

try:
    from ..utils.logger import get_logger
except ImportError: # Fallback for scenarios where the module might be run directly or structure is different
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

# Default path for service configurations if not provided explicitly.
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "agent_configs.yaml")

class APIManager:
    """
    Handles API requests, authentication, and configuration for multiple services.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the APIManager.

        Args:
            config_path: Optional path to a YAML file containing service configurations.
                         If None, attempts to load from DEFAULT_CONFIG_PATH.
        """
        self.logger = get_logger("APIManager")
        self.service_configs: Dict[str, Dict[str, Any]] = {} # Allow Any for base_url etc.
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MultiAgentSystem/1.0"})
        
        actual_config_path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        self.load_service_configs(actual_config_path)
        self.logger.info(f"APIManager initialized. Loaded configs for: {list(self.service_configs.keys())}")

    def load_service_configs(self, config_path: str):
        """
        Loads service configurations from a YAML file or environment variables.
        Prioritizes loaded config from file, then environment variables, then hardcoded defaults.
        """
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

        # Update service_configs with file_configs first
        self.service_configs.update(file_configs)

        # Define services and their expected environment variables / default structure
        services_to_configure = {
            "deepseek": {"api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com/v1"},
            "cursor": {"api_key_env": "CURSOR_API_KEY", "base_url": "https://api.cursor.ai/v1"}, # Fictional URL
            "windsurf": {"api_key_env": "WINDSURF_API_KEY", "base_url": "https://api.windsurf.ai/v1"}, # Fictional URL
            "claude": {"api_key_env": "CLAUDE_API_KEY", "base_url": "https://api.anthropic.com/v1"},
            "gemini": {"api_key_env": "GEMINI_API_KEY", "base_url": "https://generativelanguage.googleapis.com/v1beta"}
        }

        for service_name, details in services_to_configure.items():
            # If service or its essential keys are not in file_configs, try env/defaults
            if service_name not in self.service_configs or \
               "api_key" not in self.service_configs[service_name] or \
               "base_url" not in self.service_configs[service_name]:
                
                current_service_config = self.service_configs.get(service_name, {})
                
                api_key = current_service_config.get("api_key") or \
                            os.getenv(details["api_key_env"], f"YOUR_{service_name.upper()}_KEY_HERE")
                base_url = current_service_config.get("base_url") or details["base_url"]
                
                self.service_configs[service_name] = {
                    "api_key": api_key,
                    "base_url": base_url,
                    **current_service_config # Preserve other settings from file if they exist
                }
                
                if api_key == f"YOUR_{service_name.upper()}_KEY_HERE":
                    self.logger.warning(f"API key for {service_name} is a placeholder. Set {details['api_key_env']} or update config file.")
                elif service_name not in file_configs: # Only log this if it wasn't in the loaded file config
                    self.logger.info(f"Configuration for {service_name} established from environment variables/defaults.")


    def get_auth_header(self, service_name: str) -> Dict[str, str]:
        """
        Constructs the authentication header for the specified service.
        """
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("api_key"):
            self.logger.warning(f"API key not found or is empty for service: {service_name}")
            return {}

        api_key = service_info["api_key"]
        if api_key.startswith("YOUR_") and api_key.endswith("_KEY_HERE"):
             self.logger.warning(f"Using placeholder API key for {service_name} in get_auth_header.")


        if service_name in ["deepseek", "cursor", "gemini"]: # Bearer token style
            return {"Authorization": f"Bearer {api_key}"}
        elif service_name == "claude": # x-api-key style
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        elif service_name == "windsurf": # Custom token or other
            return {"X-Custom-Auth-Token": api_key} 
        
        self.logger.warning(f"Auth header style not explicitly defined for service: {service_name}. Returning empty auth header.")
        return {}

    def make_request(
        self,
        service_name: str,
        endpoint: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Makes an HTTP request to the specified service.
        """
        service_info = self.service_configs.get(service_name)
        if not service_info or not service_info.get("base_url"):
            self.logger.error(f"Base URL not configured for service: {service_name}")
            return {"error": f"Configuration missing or incomplete for {service_name}", "status_code": 500}

        base_url = service_info["base_url"]
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        auth_header = self.get_auth_header(service_name)
        headers = {"Content-Type": "application/json", **auth_header, **(extra_headers or {})}

        request_args: Dict[str, Any] = {"method": method.upper(), "url": url, "headers": headers, "timeout": timeout}

        if method.upper() in ["POST", "PUT", "PATCH"]:
            if data is not None:
                request_args["json"] = data
        elif data: # For GET, DELETE, etc. if data is provided, assume it's for params
            request_args["params"] = data 
        
        if params: # Explicit params always get precedence or are merged
            request_args["params"] = {**(request_args.get("params", {})), **params}
            
        self.logger.debug(f"Making {method.upper()} request to {url} with json_data: {request_args.get('json')}, params: {request_args.get('params')}")

        try:
            response = self.session.request(**request_args)
            self.handle_rate_limiting(service_name, response.headers)
            response.raise_for_status() 
            
            if not response.content:
                 self.logger.info(f"Received empty but successful (status {response.status_code}) response from {service_name} for endpoint {endpoint}.")
                 return {"status": "success", "data": None, "status_code": response.status_code}

            return response.json()
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error for {service_name} on {url}: {e.response.status_code} - {e.response.text[:500]}") # Log first 500 chars of error
            try:
                error_content = e.response.json()
            except json.JSONDecodeError:
                error_content = e.response.text
            return {"error": "HTTPError", "message": str(e), "status_code": e.response.status_code, "content": error_content}
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout error for {service_name} on {url} after {timeout} seconds")
            return {"error": "Timeout", "message": f"Request timed out after {timeout} seconds.", "status_code": 408}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request exception for {service_name} on {url}: {e}")
            return {"error": "RequestException", "message": str(e), "status_code": 503} # Service Unavailable for general RequestException
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response from {service_name} for {url}: {e}. Response text: {response.text[:500] if 'response' in locals() else 'N/A'}")
            return {"error": "JSONDecodeError", "message": str(e), "raw_response": response.text if 'response' in locals() else "N/A", "status_code": 502} # Bad Gateway


    def handle_rate_limiting(self, service_name: str, response_headers: Any): 
        """
        Placeholder for handling rate limiting based on response headers.
        `response_headers` is a requests.structures.CaseInsensitiveDict
        """
        remaining = response_headers.get('X-RateLimit-Remaining')
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
    
    print("  Testing GET request...")
    get_response = manager.make_request(
        service_name="public_jsonplaceholder",
        endpoint="todos/1",
        method="GET"
    )
    print(f"  GET Response: {json.dumps(get_response, indent=2)}")

    print("\n  Testing POST request...")
    post_response = manager.make_request(
        service_name="public_jsonplaceholder",
        endpoint="posts",
        method="POST",
        data={"title": "Test Post", "body": "This is a test.", "userId": 1}
    )
    print(f"  POST Response: {json.dumps(post_response, indent=2)}")

    print("\n--- APIManager testing done. ---")
    print(f"Note: A dummy config may have been created/used at {dummy_config_path}. You can remove it if desired.")
    print("To test specific service configurations, ensure their API keys are set as environment variables (e.g., DEEPSEEK_API_KEY, CLAUDE_API_KEY, etc.) or in the YAML config.")

# TODOs from original plan:
# - Consider more sophisticated API key management.
# - Implement more robust retry mechanisms (e.g., exponential backoff).
# - Expand handle_rate_limiting to actually pause/retry.
# - Securely handle and log API errors.
# - Ensure thread-safety if used in a multi-threaded context (requests.Session is not thread-safe).
#   For asyncio, aiohttp.ClientSession would be preferred.
