import unittest
from unittest.mock import MagicMock

# Assuming tests are run from the 'multi-agent-system' directory root,
# or 'multi-agent-system/src' is in PYTHONPATH.
try:
    from src.agents.deepseek_agent import DeepSeekAgent
    from tests.mocks.mock_api_manager import MockAPIManager
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.deepseek_agent import DeepSeekAgent
    from tests.mocks.mock_api_manager import MockAPIManager


class TestDeepSeekAgent(unittest.TestCase):
    def setUp(self):
        self.mock_api_manager = MockAPIManager()
        self.agent_config = {
            "model": "deepseek-test-model",
            "default_system_prompt": "You are a general test DeepSeek model.",
            "max_tokens": 1500, # Default from agent implementation
            "temperature": 0.3   # Default from agent implementation
        }
        self.agent = DeepSeekAgent(
            agent_name="TestDeepSeek",
            api_manager=self.mock_api_manager, # type: ignore
            config=self.agent_config
        )
        self.agent.logger = MagicMock()
        self.mock_api_manager.reset_mocks() # Ensure mocks are clean for each test

    def test_initialization(self):
        self.assertEqual(self.agent.get_name(), "TestDeepSeek")
        self.assertIsNotNone(self.agent.api_manager)
        self.assertEqual(self.agent.model, "deepseek-test-model")
        self.agent.logger.info.assert_any_call("DeepSeekAgent 'TestDeepSeek' initialized with model 'deepseek-test-model'.")

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("description", caps)
        self.assertIn("capabilities", caps)
        self.assertIsInstance(caps["capabilities"], list)
        self.assertTrue(len(caps["capabilities"]) > 0)
        self.assertIn("text_generation", caps["capabilities"])

    def test_process_query_success(self):
        mock_response_content = "Test response from DeepSeek"
        mock_api_response = {
            "choices": [{"message": {"content": mock_response_content}}],
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "id": "chatcmpl-xxxxx",
            "finish_reason": "stop" # Added based on DeepSeekAgent's parsing
        }
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {"prompt": "test query"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        call_args = self.mock_api_manager.make_request.call_args
        self.assertEqual(call_args[1]['service_name'], 'deepseek')
        self.assertEqual(call_args[1]['endpoint'], 'chat/completions')
        self.assertEqual(call_args[1]['method'], 'POST')

        actual_payload = call_args[1]['data']
        self.assertEqual(actual_payload['model'], self.agent_config['model'])
        self.assertEqual(len(actual_payload['messages']), 2)
        self.assertEqual(actual_payload['messages'][0]['role'], 'system')
        self.assertEqual(actual_payload['messages'][0]['content'], self.agent_config['default_system_prompt'])
        self.assertEqual(actual_payload['messages'][1]['role'], 'user')
        self.assertEqual(actual_payload['messages'][1]['content'], "test query")
        self.assertEqual(actual_payload['max_tokens'], self.agent_config['max_tokens'])
        self.assertEqual(actual_payload['temperature'], self.agent_config['temperature'])

        expected_result = {
            "status": "success",
            "content": mock_response_content,
            "finish_reason": "stop",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.info.assert_any_call(f"Successfully received and parsed response from DeepSeek for prompt: '{query_data['prompt'][:100]}...'")

    def test_process_query_api_error(self):
        error_message = "DeepSeek API unavailable"
        # This mock APIManager returns a dict that looks like an error from the APIManager itself
        self.mock_api_manager.set_make_request_response(
            response_data={"error": "APIError", "message": error_message, "status_code": 503}
        ) # No is_error=True, as this is the structure APIManager returns

        query_data = {"prompt": "test query for API error"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        expected_result = {
            "status": "error",
            "message": f"API request failed: {error_message}",
            "details": {"error": "APIError", "message": error_message, "status_code": 503}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.error.assert_any_call(f"API request failed for DeepSeek: {error_message}")

    def test_process_query_missing_prompt(self):
        query_data = {} # Empty query_data, no prompt
        result = self.agent.process_query(query_data)

        expected_result = {"status": "error", "message": "User prompt missing"}
        self.assertEqual(result, expected_result)
        self.mock_api_manager.make_request.assert_not_called()
        self.agent.logger.error.assert_any_call("User prompt is missing in query_data.")

    def test_process_query_with_custom_system_prompt_and_config_overrides(self):
        # Re-init agent with more specific config for this test, or modify existing one
        # For this test, we'll rely on setUp's agent but override via query_data
        custom_config = {
            "model": "deepseek-custom-model", # Agent's model won't change from setUp
            "temperature": 0.95,
            "max_tokens": 105,
            "default_system_prompt": "This should not be used"
        }
        # Update agent's config for this test if needed, or ensure query_data overrides work
        # self.agent.config = custom_config # This would change it for subsequent calls in this test method if not careful
        # self.agent.model = custom_config["model"] # If model needs to be different

        mock_response_content = "Custom test response"
        mock_api_response = {"choices": [{"message": {"content": mock_response_content}}], "usage": {"total_tokens": 20}, "finish_reason": "stop"}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {
            "prompt": "custom test query",
            "system_prompt": "Override system info for this query.",
            "temperature": 0.88, # Query-time override for temperature
            "max_tokens": 55    # Query-time override for max_tokens
        }
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']

        self.assertEqual(actual_payload['model'], self.agent_config['model']) # Model comes from agent's init config
        self.assertEqual(actual_payload['temperature'], 0.88) # Overridden by query_data
        self.assertEqual(actual_payload['max_tokens'], 55)    # Overridden by query_data
        self.assertEqual(actual_payload['messages'][0]['role'], 'system')
        self.assertEqual(actual_payload['messages'][0]['content'], "Override system info for this query.")
        self.assertEqual(actual_payload['messages'][1]['role'], 'user')
        self.assertEqual(actual_payload['messages'][1]['content'], "custom test query")

        expected_result = {
            "status": "success",
            "content": mock_response_content,
            "finish_reason": "stop",
            "usage": {"total_tokens": 20}
        }
        self.assertEqual(result, expected_result)

    def test_process_query_invalid_api_response_structure(self):
        # Simulate an API response that is successful (e.g. 200 OK) but has an unexpected structure
        malformed_api_response = {"unexpected_field": "some_data"} # Missing 'choices'
        self.mock_api_manager.set_make_request_response(malformed_api_response)

        query_data = {"prompt": "test query for malformed response"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid response structure from DeepSeek API", result["message"])
        self.agent.logger.error.assert_any_call(f"Failed to extract content from DeepSeek response. Response: {malformed_api_response}")

if __name__ == '__main__':
    unittest.main()
