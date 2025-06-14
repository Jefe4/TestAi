# src/agents/gemini_agent.py
"""
Specialized agent for interacting with Google's Gemini AI models.

This agent uses the `google-generativeai` SDK to communicate with Gemini models,
supporting both text and multimodal inputs. It handles API key configuration,
model instantiation, and processing of responses, including safety feedback.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union # Added Union for type hinting

try:
    from .base_agent import BaseAgent
    # APIManager is imported for constructor type consistency, though GeminiAgent uses the SDK directly for calls.
    from ..utils.api_manager import APIManager
    from ..utils.logger import get_logger
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
    from src.utils.logger import get_logger # type: ignore


# Attempt to import Google Generative AI SDK and its specific types.
# If the SDK is not installed, `genai` will be None, and the agent will fail to initialize.
try:
    import google.generativeai as genai
    # Import specific types for configuration and response handling.
    # ContentDict and GenerationConfigDict might be just Dict for older SDK versions.
    from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfigDict, ContentDict, PartType
except ImportError:
    genai = None
    HarmCategory = None
    HarmBlockThreshold = None
    GenerationConfigDict = Dict # type: ignore # Fallback type if SDK types are unavailable
    ContentDict = Dict # type: ignore        # Fallback type
    PartType = Any # type: ignore             # Fallback type
    # A warning will be logged during GeminiAgent instantiation if `genai` is None.


class GeminiAgent(BaseAgent):
    """
    An agent that utilizes Google's Gemini models for various tasks, including
    multimodal understanding (text, images), data analysis, text generation,
    and potentially function calling and chat functionalities.

    This agent directly uses the `google-generativeai` Python SDK for its operations.
    """

    def __init__(self, agent_name: str, api_manager: APIManager, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the GeminiAgent.

        This involves setting up the API key for the Google Generative AI SDK,
        configuring the generative model with specified or default parameters
        (model name, generation config, safety settings, system instructions).

        Args:
            agent_name: The user-defined name for this agent instance.
            api_manager: An instance of `APIManager`. While GeminiAgent uses the SDK directly
                         for its core API calls, APIManager might be used for other utility
                         API calls or is kept for constructor consistency across agents.
            config: Optional configuration dictionary for the agent. Expected keys:
                    - "api_key" (str, Optional): Gemini API key. If not provided, attempts to use `GEMINI_API_KEY` env var.
                    - "model" (str, Optional): The specific Gemini model to use (e.g., "gemini-1.5-flash-latest").
                    - "generation_config" (Dict, Optional): Dictionary for `genai.GenerationConfig`.
                                                            Example: `{"temperature": 0.7, "max_output_tokens": 2048}`.
                    - "safety_settings" (List[Dict], Optional): Configuration for content safety.
                                                               Example: `[{"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                                                          "threshold": HarmBlockThreshold.BLOCK_NONE}]`.
                    - "default_system_instruction" (str, Optional): A default system instruction for the model.

        Raises:
            ImportError: If the `google-generativeai` SDK is not installed.
            ValueError: If the Gemini API key is not found in config or environment variables.
            ConnectionError: If SDK configuration with the API key fails.
            RuntimeError: If the `genai.GenerativeModel` instantiation fails.
        """
        super().__init__(agent_name, config)
        self.api_manager = api_manager # Stored for potential utility use, though core calls use SDK.

        if genai is None: # Check if the SDK was imported successfully
            self.logger.error("Fatal: google-generativeai SDK is not installed. GeminiAgent cannot function.")
            raise ImportError("google-generativeai SDK not found. Please install it to use GeminiAgent (e.g., `pip install google-generativeai`).")

        # Configure API Key: from config file or environment variable.
        self.api_key: Optional[str] = self.config.get("api_key")
        if not self.api_key:
            env_api_key = os.getenv("GEMINI_API_KEY")
            if env_api_key:
                self.api_key = env_api_key
                self.logger.info("Loaded Gemini API key from GEMINI_API_KEY environment variable.")
            else:
                self.logger.error("Gemini API key not found in agent configuration or GEMINI_API_KEY environment variable.")
                raise ValueError("Gemini API key missing. Provide it in agent config (api_key) or as GEMINI_API_KEY environment variable.")
        
        try:
            genai.configure(api_key=self.api_key) # Configure the SDK globally with the API key.
        except Exception as e: # Catch potential errors during SDK configuration.
            self.logger.error(f"Failed to configure Gemini SDK with API key: {e}", exc_info=True)
            raise ConnectionError(f"Failed to configure Gemini SDK: {e}")

        # Set model name, generation config, safety settings, and system instruction.
        self.model_name: str = self.config.get("model", "gemini-1.5-flash-latest") # Default model
        
        # Default generation configuration (can be overridden by agent config)
        default_gen_config: GenerationConfigDict = {
            "temperature": 0.7,
            "top_p": 1.0,       # Nucleus sampling parameter
            "top_k": 1,         # Top-k sampling parameter
            "max_output_tokens": 2048
        }
        user_gen_config = self.config.get("generation_config", {})
        self.generation_config: GenerationConfigDict = {**default_gen_config, **user_gen_config}
        
        # Default safety settings (blocks harmful content at medium threshold)
        # These can be overridden by agent config. For less restrictive settings,
        # use HarmBlockThreshold.BLOCK_NONE for relevant categories.
        default_safety_settings: List[Dict[str, Any]] = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        self.safety_settings: List[Dict[str, Any]] = self.config.get("safety_settings", default_safety_settings)
        
        # System instruction for the model (can be None if not set)
        self.system_instruction: Optional[str] = self.config.get("default_system_instruction")

        # Instantiate the generative model from the SDK.
        try:
            self.model: genai.GenerativeModel = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self.system_instruction if self.system_instruction else None # Pass None if empty
            )
        except Exception as e: # Catch errors during model instantiation
            self.logger.error(f"Failed to instantiate Gemini GenerativeModel '{self.model_name}': {e}", exc_info=True)
            raise RuntimeError(f"Gemini model instantiation failed for '{self.model_name}': {e}")
            
        self.logger.info(f"GeminiAgent '{self.agent_name}' initialized successfully with model '{self.model_name}'.")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Describes the capabilities of the GeminiAgent.
        """
        return {
            "description": "Agent for multimodal understanding (text, images), data analysis, text generation, function calling, and chat using Google Gemini models.",
            "capabilities": ["text_generation", "multimodal_input", "data_analysis", "function_calling", "chat", "summarization"],
            "models_supported": ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro", "gemini-pro-vision"] # Example models
        }

    async def process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a query using the configured Gemini model via the SDK.

        This method constructs the prompt content (which can be multimodal) and
        makes an asynchronous call to the Gemini API using `model.generate_content_async`.
        It handles responses, including potential safety blocks and errors.

        Args:
            query_data: A dictionary containing the query details.
                        Expected keys:
                        - "prompt_parts" (List[Union[str, PartType]]): A list of content parts for the prompt.
                          For simple text, this can be `["your text query"]`.
                          For multimodal input (e.g., image and text), it could be
                          `["Describe this image:", PIL.Image.open('image.jpg')]` or
                          `["Describe this:", {"mime_type": "image/jpeg", "data": "<base64_encoded_image>"}]`.
                          The items in the list should be compatible with the `contents`
                          argument of `genai.GenerativeModel.generate_content_async`.
                        - "generation_config_override" (Optional[Dict]): Overrides the model's default
                                                                        `generation_config` for this specific query.
                        - "safety_settings_override" (Optional[List[Dict]]): Overrides the model's default
                                                                            `safety_settings` for this query.
        Returns:
            A dictionary containing:
            - "status" (str): "success" or "error".
            - "content" (str, optional): The primary textual content of Gemini's response, if successful and applicable.
            - "full_response_parts_texts" (List[str], optional): List of texts from all parts of the response.
            - "message" (str, optional): An error message, if an error occurred or content was blocked.
            - "details" (str, optional): Additional details about errors or blocking.
            - "finish_reason" (str, optional): The reason the model stopped generating (e.g., "STOP", "MAX_TOKENS", "SAFETY").
            - "safety_ratings" (List[str], optional): Safety ratings for the response candidates.
            - "raw_response_obj_str" (str, optional): A truncated string representation of the raw SDK response object for debugging.
        """
        if genai is None: # Should have been caught in __init__, but as a safeguard
             self.logger.error("Gemini SDK not available for process_query. This should not happen if initialization succeeded.")
             return {"status": "error", "message": "Gemini SDK not installed or failed to import."}

        user_prompt_parts: Optional[List[Any]] = query_data.get("prompt_parts")
        # Validate prompt_parts: must be a list and the first part should not be empty/None (basic check)
        if not user_prompt_parts or not isinstance(user_prompt_parts, list) or not user_prompt_parts[0]:
            self.logger.error("User 'prompt_parts' are missing, not a list, or the first part is empty in query_data for GeminiAgent.")
            return {"status": "error", "message": "Valid 'prompt_parts' (list with non-empty first part) missing."}

        # Get overrides for generation config and safety settings for this specific query
        generation_config_override = query_data.get("generation_config_override")
        safety_settings_override = query_data.get("safety_settings_override")

        # Log a summary of the prompt being sent
        log_prompt_summary_parts = []
        for part in user_prompt_parts:
            if isinstance(part, str):
                log_prompt_summary_parts.append(part[:100] + "..." if len(part) > 100 else part)
            elif hasattr(part, 'mime_type'): # For SDK PartType objects or dicts with mime_type
                log_prompt_summary_parts.append(f"<MediaPart: {getattr(part, 'mime_type', 'unknown_mime_type')}>")
            else: # Fallback for other types (e.g., PIL Images)
                log_prompt_summary_parts.append(f"<{type(part).__name__}>")
        self.logger.info(f"Sending query to Gemini model '{self.model_name}'. Prompt summary: {', '.join(log_prompt_summary_parts)}")

        try:
            # Determine effective generation_config and safety_settings for this call
            current_gen_config = generation_config_override if generation_config_override else self.model.generation_config
            current_safety_settings = safety_settings_override if safety_settings_override else self.model.safety_settings
            
            # Make the asynchronous API call using the SDK
            response = await self.model.generate_content_async(
                contents=user_prompt_parts,       # Prompt content (list of parts)
                generation_config=current_gen_config,
                safety_settings=current_safety_settings
                # stream=False is the default for generate_content_async
            )
            
            # Check for prompt feedback indicating blocking *before* trying to access `response.text`.
            # `response.prompt_feedback` exists if the prompt itself was blocked.
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason_str = response.prompt_feedback.block_reason.name # Get enum name
                self.logger.warning(f"Gemini prompt was blocked due to: {block_reason_str}. "
                                    f"Safety ratings: {response.prompt_feedback.safety_ratings}")
                return {
                    "status": "error", 
                    "message": f"Prompt blocked by Gemini due to {block_reason_str}",
                    "details": str(response.prompt_feedback) # String representation of PromptFeedback object
                }

            # If the prompt was not blocked, try to access `response.text`.
            # This can raise a ValueError if the *generated content* was blocked (e.g., due to safety settings on candidates).
            extracted_text: str
            try:
                extracted_text = response.text # Concatenates text from all parts, if available and not blocked.
            except ValueError as ve: # This ValueError often indicates that content in candidates was blocked.
                 self.logger.warning(f"Gemini content generation potentially blocked or response empty. SDK raised ValueError: {ve}")
                 # Inspect candidates for more detailed blocking reasons
                 blocked_candidate_details: List[str] = []
                 if response.candidates: # Check if there are candidates
                     for cand_idx, cand in enumerate(response.candidates):
                         if cand.finish_reason != genai.types.FinishReason.STOP: # Check if not a normal stop
                             reason_name = cand.finish_reason.name if cand.finish_reason else "UNKNOWN_REASON"
                             ratings_str = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in cand.safety_ratings])
                             blocked_candidate_details.append(
                                 f"Candidate {cand_idx} finish_reason: {reason_name}. SafetyRatings: [{ratings_str}]"
                             )
                 return {
                     "status": "error",
                     "message": f"Content generation issue or blocked content: {ve}",
                     "details": "; ".join(blocked_candidate_details) if blocked_candidate_details else "No specific block details found in candidates, or response was empty."
                 }

            self.logger.info(f"Successfully received response from Gemini model '{self.model_name}'.")
            
            # Extract text from all parts of the response for more detailed output if needed.
            response_parts_texts: List[str] = []
            if response.parts: # `response.parts` gives access to individual content parts
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        response_parts_texts.append(part.text)
                    elif hasattr(part, 'function_call'): # Handle function calls if they occur
                        response_parts_texts.append(f"FunctionCall: name='{part.function_call.name}', args={part.function_call.args}")
                    # Could add handling for other part types like 'blob' if expecting file data.
            
            # Determine finish reason and safety ratings from the first candidate (most relevant one).
            finish_reason_name = "UNKNOWN"
            safety_ratings_str_list: List[str] = []
            if response.candidates: # Response should have candidates
                first_candidate = response.candidates[0]
                finish_reason_name = first_candidate.finish_reason.name if first_candidate.finish_reason else "UNKNOWN"
                if first_candidate.safety_ratings:
                    safety_ratings_str_list = [f"{sr.category.name}: {sr.probability.name}" for sr in first_candidate.safety_ratings]

            return {
                "status": "success",
                "content": extracted_text,  # Primary combined textual content
                "full_response_parts_texts": response_parts_texts, # List of texts from all response parts
                "finish_reason": finish_reason_name,
                "safety_ratings": safety_ratings_str_list,
                "raw_response_obj_str": str(response)[:500] # Truncated string of raw SDK response object for debugging
            }

        except Exception as e: # Catch any other unexpected errors during the SDK call or processing.
            self.logger.error(f"Gemini API call or response processing failed: {type(e).__name__} - {e}", exc_info=True)
            # Try to extract a more specific error message if it's a Google API core exception
            error_message = str(e)
            if hasattr(e, 'message') and isinstance(getattr(e, 'message'), str): # Common for google.api_core.exceptions
                error_message = getattr(e, 'message')
            return {"status": "error", "message": error_message}


if __name__ == '__main__':
    # This block is for basic demonstration and testing.
    # WARNING: Running this directly will attempt to make real API calls if GEMINI_API_KEY is set.
    # For true unit testing, genai.GenerativeModel should be mocked.

async def main_gemini_test(): # Wrapped in async main function
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
                dummy_api_manager_for_constructor = APIManager() if 'APIManager' in globals() and APIManager is not None else None # type: ignore
                if dummy_api_manager_for_constructor:
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
                    print("APIManager not available for GeminiAgent instantiation test (likely fallback import scenario).")
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
        dummy_api_manager = APIManager() if 'APIManager' in globals() and APIManager is not None else None # type: ignore

        if not dummy_api_manager:
            print("APIManager not available for live test, skipping.")
            return

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
            response1 = await gemini_agent.process_query(query1_data) # Awaited
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
            response2 = await gemini_agent.process_query(query2_data) # Awaited
            print(f"Response 2: {response2}")
            assert response2["status"] == "error"
            assert "Valid 'prompt_parts' (list) missing or empty" in response2["message"]
            print("Test Case 2 Passed (correctly handled missing prompt).")

            # Test case 3: Query that might be blocked by safety (example)
            print("\n--- Test Case 3: Potentially Safety-Blocked Query (Illustrative) ---")
            print("Skipping Test Case 3 (safety blocking) as results are unpredictable without fine-tuned inputs/settings.")


        except Exception as e:
            print(f"Error during GeminiAgent live testing: {e}")
            print("Ensure your GEMINI_API_KEY is valid and has access to the specified model.")

    print("\n--- GeminiAgent testing block finished. ---")
    if genai is None:
        print("Reminder: google-generativeai SDK is not installed. GeminiAgent functionality is unavailable.")

if __name__ == '__main__':
    # import asyncio # Added at top
    if os.name == 'nt': # Optional: Windows specific policy for asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_gemini_test())

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
