import unittest
import os
from unittest.mock import patch, MagicMock, PropertyMock, AsyncMock # Added AsyncMock

# Attempt to import google.generativeai.types for type hinting and constants in tests
# The actual 'genai' module used by the agent will be mocked.
try:
    import google.generativeai.types as genai_types 
    from google.generativeai.types import HarmCategory, HarmBlockThreshold # For convenience
    SDK_AVAILABLE = True
except ImportError:
    # Define dummy types if SDK is not available, so tests can still be structured
    # The agent itself handles the absence of the SDK at runtime.
    class DummyHarmCategory: HARM_CATEGORY_HARASSMENT = "HARM_CATEGORY_HARASSMENT"; HARM_CATEGORY_HATE_SPEECH = "HARM_CATEGORY_HATE_SPEECH"; HARM_CATEGORY_SEXUALLY_EXPLICIT = "HARM_CATEGORY_SEXUALLY_EXPLICIT"; HARM_CATEGORY_DANGEROUS_CONTENT = "HARM_CATEGORY_DANGEROUS_CONTENT"; HARM_CATEGORY_SEXUAL = "HARM_CATEGORY_SEXUAL" # Add others if used
    class DummyHarmBlockThreshold: BLOCK_MEDIUM_AND_ABOVE = "BLOCK_MEDIUM_AND_ABOVE"; BLOCK_NONE = "BLOCK_NONE" # Add others if used
    genai_types = MagicMock() # Mock the whole types module if not available
    genai_types.HarmCategory = DummyHarmCategory
    genai_types.HarmBlockThreshold = DummyHarmBlockThreshold
    HarmCategory = DummyHarmCategory # Make it directly available
    HarmBlockThreshold = DummyHarmBlockThreshold # Make it directly available
    SDK_AVAILABLE = False


# Assuming tests are run from the 'multi-agent-system' directory root
try:
    from src.agents.gemini_agent import GeminiAgent
    # APIManager is needed for constructor, but not directly used by GeminiAgent for calls
    from src.utils.api_manager import APIManager 
except ImportError:
    # Fallback for different execution contexts
    import sys
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
    from src.agents.gemini_agent import GeminiAgent
    from src.utils.api_manager import APIManager


class TestGeminiAgent(unittest.IsolatedAsyncioTestCase): # Changed inheritance
    def setUp(self):
        self.agent_config = {
            "api_key": "TEST_GEMINI_KEY",
            "model": "gemini-test-model",
            "generation_config": {"temperature": 0.5, "max_output_tokens": 100},
            "safety_settings": [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
            ],
            "default_system_instruction": "You are a test Gemini model."
        }

        # Mock the 'genai' module that the GeminiAgent imports
        self.mock_genai_module = MagicMock()
        
        # Mock the GenerativeModel class and its instance
        self.mock_generative_model_instance = MagicMock()
        # Key change: mock the async method specifically
        self.mock_generative_model_instance.generate_content_async = AsyncMock() 
        self.mock_genai_module.GenerativeModel.return_value = self.mock_generative_model_instance
        
        # Mock the configure function
        self.mock_genai_module.configure = MagicMock()

        # Mock HarmCategory and HarmBlockThreshold if SDK was available, otherwise they are dummies
        self.mock_genai_module.types = MagicMock()
        if SDK_AVAILABLE:
            self.mock_genai_module.types.HarmCategory = genai_types.HarmCategory
            self.mock_genai_module.types.HarmBlockThreshold = genai_types.HarmBlockThreshold
        else: # Use the dummy ones if SDK is not installed
            self.mock_genai_module.types.HarmCategory = DummyHarmCategory
            self.mock_genai_module.types.HarmBlockThreshold = DummyHarmBlockThreshold


        self.genai_patcher = patch('src.agents.gemini_agent.genai', self.mock_genai_module)
        self.mock_genai_module_for_agent = self.genai_patcher.start()
        
        # Mock APIManager, though not used for core functionality by GeminiAgent
        self.mock_api_manager = MagicMock(spec=APIManager)

        self.agent = GeminiAgent(
            agent_name="TestGemini",
            api_manager=self.mock_api_manager, 
            config=self.agent_config
        )
        self.agent.logger = MagicMock() # Suppress logging output

    def tearDown(self):
        self.genai_patcher.stop()

    def test_initialization(self):
        self.assertEqual(self.agent.get_name(), "TestGemini")
        self.mock_genai_module_for_agent.configure.assert_called_once_with(api_key="TEST_GEMINI_KEY")
        
        expected_safety_settings_for_sdk = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
        ]
        
        self.mock_genai_module_for_agent.GenerativeModel.assert_called_once_with(
            model_name="gemini-test-model",
            generation_config=self.agent_config["generation_config"],
            safety_settings=expected_safety_settings_for_sdk,
            system_instruction=self.agent_config["default_system_instruction"]
        )
        self.agent.logger.info.assert_any_call("GeminiAgent 'TestGemini' initialized with model 'gemini-test-model'.")

    @patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True) # Ensure no env var fallback
    def test_initialization_missing_api_key(self):
        config_no_key = self.agent_config.copy()
        del config_no_key["api_key"]
        with self.assertRaisesRegex(ValueError, "Gemini API key missing"):
            GeminiAgent(
                agent_name="TestGeminiNoKey",
                api_manager=self.mock_api_manager,
                config=config_no_key
            )
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": "env_gemini_key"}, clear=True)
    def test_initialization_api_key_from_env(self):
        config_no_key = self.agent_config.copy()
        del config_no_key["api_key"]
        
        agent_env_key = GeminiAgent(
            agent_name="TestGeminiEnvKey",
            api_manager=self.mock_api_manager,
            config=config_no_key
        )
        self.mock_genai_module_for_agent.configure.assert_called_with(api_key="env_gemini_key")
        self.assertEqual(agent_env_key.api_key, "env_gemini_key")


    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("description", caps)
        self.assertIn("capabilities", caps)
        self.assertIsInstance(caps["capabilities"], list)

    async def test_process_query_success(self): # Async
        mock_response = MagicMock()
        type(mock_response).text = PropertyMock(return_value="Test response from Gemini")
        mock_response.parts = [MagicMock(text="Test response from Gemini")]
        mock_response.prompt_feedback = None 
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"), safety_ratings=[])]
        self.mock_generative_model_instance.generate_content_async.return_value = mock_response # Use async mock
        
        query_data = {"prompt_parts": ["test gemini query"]}
        result = await self.agent.process_query(query_data) # Await

        self.mock_generative_model_instance.generate_content_async.assert_awaited_once_with( # Awaited
            contents=query_data["prompt_parts"],
            generation_config=self.agent.generation_config, 
            safety_settings=self.agent.safety_settings 
        )
        
        expected_result_subset = {
            "status": "success", 
            "content": "Test response from Gemini",
            "finish_reason": "STOP"
        }
        self.assertDictContainsSubset(expected_result_subset, result)
        self.agent.logger.info.assert_any_call("Successfully received response from Gemini.")

    async def test_process_query_sdk_error(self): # Async
        self.mock_generative_model_instance.generate_content_async.side_effect = Exception("Gemini SDK Internal Error") # Use async mock
        
        query_data = {"prompt_parts": ["test query for SDK error"]}
        result = await self.agent.process_query(query_data) # Await
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Gemini SDK Internal Error")
        self.agent.logger.error.assert_any_call("Gemini API call failed: Exception - Gemini SDK Internal Error")

    async def test_process_query_missing_prompt_parts(self): # Async
        result_empty = await self.agent.process_query({}) # Await
        self.assertEqual(result_empty["status"], "error")
        self.assertIn("Valid 'prompt_parts' (list) missing or empty", result_empty["message"])
        
        result_none = await self.agent.process_query({"prompt_parts": None}) # Await
        self.assertEqual(result_none["status"], "error")
        self.assertIn("Valid 'prompt_parts' (list) missing or empty", result_none["message"])

        result_empty_list = await self.agent.process_query({"prompt_parts": []}) # Await
        self.assertEqual(result_empty_list["status"], "error")
        self.assertIn("Valid 'prompt_parts' (list) missing or empty", result_empty_list["message"])

        self.mock_generative_model_instance.generate_content_async.assert_not_called() # Use async mock

    async def test_process_query_with_dynamic_gen_config_and_safety(self): # Async
        mock_response = MagicMock()
        type(mock_response).text = PropertyMock(return_value="Dynamic config response")
        mock_response.parts = [MagicMock(text="Dynamic config response")]
        mock_response.prompt_feedback = None
        mock_response.candidates = [MagicMock(finish_reason=MagicMock(name="STOP"), safety_ratings=[])]
        self.mock_generative_model_instance.generate_content_async.return_value = mock_response # Use async mock

        dynamic_gen_config = {"temperature": 0.1, "max_output_tokens": 50}
        dynamic_safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_SEXUAL, "threshold": HarmBlockThreshold.BLOCK_NONE}
        ]
        
        query_data = {
            "prompt_parts": ["query with dynamic settings"], 
            "generation_config_override": dynamic_gen_config,
            "safety_settings_override": dynamic_safety_settings
        }
        result = await self.agent.process_query(query_data) # Await

        self.mock_generative_model_instance.generate_content_async.assert_awaited_once_with( # Awaited
            contents=query_data["prompt_parts"],
            generation_config=dynamic_gen_config, 
            safety_settings=dynamic_safety_settings 
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], "Dynamic config response")

    async def test_process_query_prompt_blocked(self): # Async
        mock_response = MagicMock()
        mock_feedback = MagicMock()
        mock_feedback.block_reason = "SAFETY"
        type(mock_response).prompt_feedback = PropertyMock(return_value=mock_feedback)
        self.mock_generative_model_instance.generate_content_async.return_value = mock_response # Use async mock

        query_data = {"prompt_parts": ["a problematic prompt"]}
        result = await self.agent.process_query(query_data) # Await

        self.assertEqual(result["status"], "error")
        self.assertIn("Prompt blocked by Gemini due to SAFETY", result["message"])
        self.agent.logger.warning.assert_any_call("Gemini prompt blocked due to: SAFETY")
        
    async def test_process_query_content_blocked_in_candidate(self): # Async
        mock_response = MagicMock()
        type(mock_response).text = PropertyMock(side_effect=ValueError("Content blocked")) 
        mock_response.parts = [] 
        mock_response.prompt_feedback = None 
        
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = MagicMock(name="SAFETY") 
        mock_candidate.safety_ratings = [MagicMock(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, probability="HIGH")]
        mock_response.candidates = [mock_candidate]
        
        self.mock_generative_model_instance.generate_content_async.return_value = mock_response # Use async mock

        query_data = {"prompt_parts": ["a query that generates a blocked response"]}
        result = await self.agent.process_query(query_data) # Await
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Content generation issue: Content blocked", result["message"])
        self.assertIn("Candidate finish reason: SAFETY", result["details"])
        self.agent.logger.warning.assert_any_call("Gemini content generation potentially blocked or empty: Content blocked")


if __name__ == '__main__':
    # This allows running the tests directly from this file
    # It's useful if the google-generativeai SDK is available in the environment
    if not SDK_AVAILABLE:
        print("WARNING: google-generativeai SDK not found. Some type checks in tests might use dummies.")
        print("GeminiAgent itself will raise an ImportError if run without the SDK.")
    unittest.main()
