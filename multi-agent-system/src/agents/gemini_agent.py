# src/agents/gemini_agent.py
"""Specialized agent for interacting with Google's Gemini AI models."""

from typing import Dict, Any, Optional, List

try:
    from .base_agent import BaseAgent
    from ..utils.api_manager import APIManager # For constructor consistency, though not directly used for API calls
    from ..utils.logger import get_logger # BaseAgent provides self.logger, but direct import can be fallback
except ImportError:
    # Fallback for direct script execution or import issues
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.base_agent import BaseAgent # type: ignore
    from src.utils.api_manager import APIManager # type: ignore
    from src.utils.logger import get_logger # type: ignore


# Attempt to import Google Generative AI SDK
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfigDict, ContentDict
    # For older versions, ContentDict might not exist, parts are just dicts.
except ImportError:
    genai = None # type: ignore 
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    GenerationConfigDict = Dict # type: ignore # Fallback type
    ContentDict = Dict # type: ignore # Fallback type
    # Log this issue if logger is available at module load time (it isn't easily)
    # print("ERROR: google-generativeai SDK not installed. GeminiAgent will not function.")


class GeminiAgent(BaseAgent):
    """
    An agent that utilizes Google's Gemini models for multimodal understanding,
    data analysis, text generation, and other advanced tasks.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the GeminiAgent.

        Args:
            agent_name: The name of the agent.
            api_manager: An instance of APIManager (kept for constructor consistency).
            config: Optional configuration dictionary for the agent.
                    Expected keys: "api_key", "model", "generation_config", 
                                   "safety_settings", "default_system_instruction".
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager # Stored for consistency, not used for SDK calls

        if genai is None:
            self.logger.error("google-generativeai SDK is not installed. GeminiAgent cannot function.")
            raise ImportError("google-generativeai SDK not found. Please install it to use GeminiAgent.")

        self.api_key = self.config.get("api_key")
        if not self.api_key:
            env_api_key = os.getenv("GEMINI_API_KEY")
            if env_api_key:
                self.api_key = env_api_key
                self.logger.info("Loaded Gemini API key from GEMINI_API_KEY environment variable.")
            else:
                self.logger.error("Gemini API key not found in config or GEMINI_API_KEY environment variable.")
                raise ValueError("Gemini API key missing. Provide it in agent config or as GEMINI_API_KEY env variable.")
        
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini SDK: {e}")
            raise ConnectionError(f"Failed to configure Gemini SDK: {e}")

        self.model_name = self.config.get("model", "gemini-1.5-flash-latest")
        
        # Default generation config
        default_gen_config: GenerationConfigDict = { # type: ignore
            "temperature": 0.7, "top_p": 1.0, "top_k": 1, "max_output_tokens": 2048 
        }
        # Merge with user-provided config
        user_gen_config = self.config.get("generation_config", {})
        self.generation_config: GenerationConfigDict = {**default_gen_config, **user_gen_config} # type: ignore
        
        # Default safety settings
        default_safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        self.safety_settings = self.config.get("safety_settings", default_safety_settings)
        
        self.system_instruction = self.config.get("default_system_instruction") # Can be None

        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config, # type: ignore
                safety_settings=self.safety_settings, # type: ignore
                system_instruction=self.system_instruction if self.system_instruction else None 
            )
        except Exception as e:
            self.logger.error(f"Failed to instantiate Gemini GenerativeModel: {e}")
            raise RuntimeError(f"Gemini model instantiation failed: {e}")
            
        self.logger.info(f"GeminiAgent '{self.agent_name}' initialized with model '{self.model_name}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the GeminiAgent.
        """
        return {
            "description": "Agent for multimodal understanding, data analysis, and general queries using Google Gemini.",
            "capabilities": ["text_generation", "multimodal_input", "data_analysis", "function_calling", "chat"],
            "models_supported": ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"] 
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]: # Changed to async
        """
        Processes a query using the Gemini API.

        Args:
            query_data: A dictionary containing the query details.
                        Expected keys:
                        - "prompt_parts" (List[Any]): A list of content parts for the prompt.
                          For simple text, this can be `["your text query"]`.
                          For multimodal, it can be `["text part", PIL.Image.open('image.jpg')]`.
                        - "generation_config_override" (Optional[Dict]): Override model's generation config.
                        - "safety_settings_override" (Optional[List[Dict]]): Override model's safety settings.
        Returns:
            A dictionary containing the status of the operation and the response content or error message.
        """
        if genai is None: # Should have been caught in __init__ but good to double check
             self.logger.error("Gemini SDK not available for process_query.")
             return {"status": "error", "message": "Gemini SDK not installed."}

        user_prompt_parts = query_data.get("prompt_parts")
        if not user_prompt_parts or not isinstance(user_prompt_parts, list) or not user_prompt_parts[0]:
            self.logger.error("User 'prompt_parts' are missing, not a list, or empty in query_data.")
            return {"status": "error", "message": "Valid 'prompt_parts' (list) missing or empty."}

        generation_config_override = query_data.get("generation_config_override")
        safety_settings_override = query_data.get("safety_settings_override")

        # For logging, convert parts to string carefully
        log_prompt = []
        for part in user_prompt_parts:
            if isinstance(part, str):
                log_prompt.append(part[:100] + "..." if len(part) > 100 else part)
            else:
                log_prompt.append(f"<{type(part).__name__}>")
        self.logger.info(f"Sending query to Gemini ({self.model_name}): {', '.join(log_prompt)}")

        try:
            # Use current model's settings unless overrides are provided
            current_gen_config = generation_config_override if generation_config_override else self.model.generation_config
            current_safety_settings = safety_settings_override if safety_settings_override else self.model.safety_settings
            
            # Use generate_content_async for asynchronous operation
            response = await self.model.generate_content_async( # Await the async call
                contents=user_prompt_parts, # type: ignore
                generation_config=current_gen_config, # type: ignore
                safety_settings=current_safety_settings # type: ignore
                # stream=False by default
            )
            
            # response.text provides concatenated text from all parts if successful
            # If there's a finish_reason like SAFETY, response.text might raise an error.
            # It's safer to check prompt_feedback first.
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                self.logger.warning(f"Gemini prompt blocked due to: {response.prompt_feedback.block_reason}")
                return {
                    "status": "error", 
                    "message": f"Prompt blocked by Gemini due to {response.prompt_feedback.block_reason}",
                    "details": str(response.prompt_feedback)
                }

            # If not blocked, try to access response.text
            # This might fail if the response was stopped for safety reasons related to *candidates*
            try:
                extracted_text = response.text
            except ValueError as ve: # Often indicates content blocked in candidates
                 self.logger.warning(f"Gemini content generation potentially blocked or empty: {ve}")
                 # Check candidates for details
                 blocked_details = []
                 for cand in response.candidates:
                     if cand.finish_reason.name != "STOP": # Not a normal stop
                         blocked_details.append(f"Candidate finish reason: {cand.finish_reason.name}, Safety Ratings: {cand.safety_ratings}")
                 return {
                     "status": "error",
                     "message": f"Content generation issue: {ve}",
                     "details": "; ".join(blocked_details) if blocked_details else "No specific block details in candidates."
                 }


            self.logger.info(f"Successfully received response from Gemini.")
            
            # Construct a more detailed response if needed
            response_parts_str = []
            if response.parts:
                for part in response.parts:
                    if hasattr(part, 'text'):
                        response_parts_str.append(part.text)
                    elif hasattr(part, 'function_call'):
                        response_parts_str.append(f"FunctionCall: {part.function_call.name}")
                    # Add more part types as needed (e.g., blob for file data)
            
            return {
                "status": "success",
                "content": extracted_text, # Main textual content
                "full_response_parts_texts": response_parts_str, # List of texts from parts
                "finish_reason": response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN",
                "safety_ratings": [str(sr) for sr in response.candidates[0].safety_ratings] if response.candidates else [],
                "raw_response_obj_str": str(response)[:500] # For debugging, truncated
            }
        except Exception as e:
            self.logger.error(f"Gemini API call failed: {type(e).__name__} - {e}")
            # Attempt to get more specific error details if it's a Google API error
            if hasattr(e, 'message'): # google.api_core.exceptions often have a message
                error_message = getattr(e, 'message')
            else:
                error_message = str(e)
            return {"status": "error", "message": error_message}


if __name__ == '__main__':
    # This block is for basic demonstration and testing.
    # WARNING: Running this directly will attempt to make real API calls if GEMINI_API_KEY is set.
    # For true unit testing, genai.GenerativeModel should be mocked.

    # Basic logger for testing if not run as part of the full system
    try:
        logger = get_logger("GeminiAgentTest") # type: ignore
    except NameError: # If get_logger itself isn't defined due to import fallbacks
        import logging
        logger = logging.getLogger("GeminiAgentTest")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)


    print("--- Testing GeminiAgent ---")

    # Attempt to get API key from environment for this test
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("GEMINI_API_KEY environment variable not set. Skipping live API tests.")
        print("GeminiAgent basic instantiation test will proceed if SDK is installed.")
        # Try to instantiate to check class structure, even if it fails at API key step
        try:
            if genai: # Check if SDK was imported
                agent_config_no_key = {
                    "model": "gemini-1.5-flash-latest",
                    "api_key": "DUMMY_KEY_FOR_INIT_TEST_ONLY" # Will fail if genai.configure is strict
                }
                # This dummy APIManager won't be used by GeminiAgent for calls
                dummy_api_manager_for_constructor = APIManager() # type: ignore
                try:
                    gemini_agent_test_init = GeminiAgent(
                        agent_name="TestGeminiNoKey",
                        api_manager=dummy_api_manager_for_constructor, # type: ignore
                        config=agent_config_no_key
                    )
                    print(f"Agent Name: {gemini_agent_test_init.get_name()}")
                    print(f"Agent Capabilities: {gemini_agent_test_init.get_capabilities()}")
                    print("GeminiAgent instantiated with dummy key (SDK configure likely failed or used dummy).")
                except Exception as e:
                    print(f"GeminiAgent instantiation failed as expected without valid key setup: {e}")
            else:
                print("Gemini SDK (google-generativeai) not installed. Cannot test GeminiAgent.")
        except Exception as e:
            print(f"Error during basic GeminiAgent instantiation test: {e}")

    else:
        print("GEMINI_API_KEY found. Attempting to initialize GeminiAgent for a simple query.")
        agent_config_with_key = {
            "model": "gemini-1.5-flash-latest", # Use a fast and common model for testing
            "api_key": gemini_api_key, # This will be used by genai.configure()
             "generation_config": {"temperature": 0.5, "max_output_tokens": 100},
             "default_system_instruction": "You are a concise assistant."
        }
        
        # Dummy APIManager for constructor consistency
        dummy_api_manager = APIManager() if 'APIManager' in globals() else None # type: ignore

        try:
            gemini_agent = GeminiAgent(
                agent_name="TestGeminiAgentLive",
                api_manager=dummy_api_manager, # type: ignore
                config=agent_config_with_key
            )

            print(f"Agent Name: {gemini_agent.get_name()}")
            print(f"Agent Capabilities: {gemini_agent.get_capabilities()}")
            print(f"Agent Configured Model: {gemini_agent.model_name}")

            # Test case 1: Simple text query
            print("\n--- Test Case 1: Simple Text Query ---")
            query1_data = {"prompt_parts": ["What is the main component of the sun?"]}
            response1 = gemini_agent.process_query(query1_data)
            print(f"Response 1: {response1}")
            if response1["status"] == "success":
                assert "hydrogen" in response1.get("content", "").lower() or \
                       "plasma" in response1.get("content", "").lower()
                print("Test Case 1 Passed (content seems plausible).")
            else:
                print(f"Test Case 1 Failed: {response1.get('message')}")

            # Test case 2: Missing prompt_parts
            print("\n--- Test Case 2: Missing prompt_parts ---")
            query2_data = {}
            response2 = gemini_agent.process_query(query2_data)
            print(f"Response 2: {response2}")
            assert response2["status"] == "error"
            assert "Valid 'prompt_parts' (list) missing or empty" in response2["message"]
            print("Test Case 2 Passed (correctly handled missing prompt).")

            # Test case 3: Query that might be blocked by safety (example)
            # This is highly dependent on default and actual safety settings of the API key / model version
            print("\n--- Test Case 3: Potentially Safety-Blocked Query (Illustrative) ---")
            # query3_data = {"prompt_parts": ["Tell me how to do something very dangerous."]} # Example
            # response3 = gemini_agent.process_query(query3_data)
            # print(f"Response 3: {response3}")
            # if response3["status"] == "error" and "blocked" in response3.get("message","").lower():
            #    print("Test Case 3 Passed (query was blocked as expected or similar error).")
            # elif response3["status"] == "success":
            #    print("Test Case 3 Warning: Query was not blocked. Content: ", response3.get("content","")[:100] + "...")
            # else:
            #    print(f"Test Case 3 Result: {response3.get('message')}")
            print("Skipping Test Case 3 (safety blocking) as results are unpredictable without fine-tuned inputs/settings.")


        except Exception as e:
            print(f"Error during GeminiAgent live testing: {e}")
            print("Ensure your GEMINI_API_KEY is valid and has access to the specified model.")

    print("\n--- GeminiAgent testing block finished. ---")
    if genai is None:
        print("Reminder: google-generativeai SDK is not installed. GeminiAgent functionality is unavailable.")

# TODO:
# - Add support for multimodal inputs (images, etc.) in prompt_parts.
# - Implement chat history management if using `start_chat()`.
# - Expose more generation parameters (e.g., stream, stop_sequences).
# - More granular error handling for specific Gemini API errors.
# - For function calling, `generate_content` response needs to check for `response.function_calls`.
# - Consider how `system_instruction` should be best handled (model init vs. dynamic per call, if API allows).
#   The new `genai.GenerativeModel(system_instruction=...)` is the preferred way for system-level prompts.
#   If dynamic system prompts are needed per call, they should be part of the `contents` list with `role: 'system'`.
#   The current `process_query` assumes `user_prompt_parts` is the primary content.
#   If `query_data` includes a `system_instruction`, it might need to be prepended to `user_prompt_parts` as a 'system' role message,
#   or the model re-initialized if that's the only way the SDK supports it for `generate_content`.
#   However, `genai.GenerativeModel` has `system_instruction` at init, so this agent uses that.
#   Dynamic system instructions per call for `generate_content` are not directly supported via a separate parameter;
#   they would need to be part of the `contents`.
#   The current code uses `self.system_instruction` from config at model init.
#   If `query_data` were to provide a `system_instruction`, this agent isn't currently set up to re-init the model or use it dynamically in `generate_content`
#   unless it's manually added to `user_prompt_parts` by the caller.
