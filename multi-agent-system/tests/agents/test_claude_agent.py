import unittest
from unittest.mock import MagicMock

# Assuming tests are run from the 'multi-agent-system' directory root
try:
    from src.agents.claude_agent import ClaudeAgent
    from tests.mocks.mock_api_manager import MockAPIManager
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.claude_agent import ClaudeAgent
    from tests.mocks.mock_api_manager import MockAPIManager

class TestClaudeAgent(unittest.TestCase):
    def setUp(self):
        self.mock_api_manager = MockAPIManager()
        self.agent_config = {
            "model": "claude-test-model",
            "default_system_prompt": "You are a general test Claude model.",
            "max_tokens": 1000, # Default from agent implementation (or typical value)
            "temperature": 0.5   # Default from agent implementation
        }
        self.agent = ClaudeAgent(
            agent_name="TestClaude",
            api_manager=self.mock_api_manager, # type: ignore
            config=self.agent_config
        )
        self.agent.logger = MagicMock()
        self.mock_api_manager.reset_mocks() # Ensure mocks are clean for each test

    def test_initialization(self):
        self.assertEqual(self.agent.get_name(), "TestClaude")
        self.assertIsNotNone(self.agent.api_manager)
        self.assertEqual(self.agent.model, "claude-test-model")
        self.agent.logger.info.assert_any_call("ClaudeAgent 'TestClaude' initialized with model 'claude-test-model'.")

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("description", caps)
        self.assertIn("capabilities", caps)
        self.assertIsInstance(caps["capabilities"], list)
        self.assertTrue(len(caps["capabilities"]) > 0)
        self.assertIn("text_generation", caps["capabilities"])

    def test_process_query_success(self):
        mock_response_text = "Test response from Claude"
        mock_api_response = {
            "content": [{"type": "text", "text": mock_response_text}],
            "model": self.agent_config["model"],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test query"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        call_args = self.mock_api_manager.make_request.call_args
        self.assertEqual(call_args[1]['service_name'], 'claude')
        self.assertEqual(call_args[1]['endpoint'], 'messages')
        self.assertEqual(call_args[1]['method'], 'POST')
        
        actual_payload = call_args[1]['data']
        self.assertEqual(actual_payload['model'], self.agent_config['model'])
        self.assertEqual(len(actual_payload['messages']), 1)
        self.assertEqual(actual_payload['messages'][0]['role'], 'user')
        self.assertEqual(actual_payload['messages'][0]['content'], "test query")
        self.assertNotIn("system", actual_payload) # No system prompt provided in query
        self.assertEqual(actual_payload['max_tokens'], self.agent_config['max_tokens']) # Check if it uses max_tokens_to_sample
        self.assertEqual(actual_payload['temperature'], self.agent_config['temperature'])

        expected_result = {
            "status": "success", 
            "content": mock_response_text,
            "finish_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.info.assert_any_call(f"Successfully received and parsed response from Claude for prompt: '{query_data['prompt'][:100]}...'")

    def test_process_query_success_with_system_prompt(self):
        mock_response_text = "System-guided response"
        mock_api_response = {"content": [{"type": "text", "text": mock_response_text}], "stop_reason": "end_turn"}
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        system_instructions = "System instructions for Claude"
        query_data = {"prompt": "test query with system prompt", "system_prompt": system_instructions}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']
        
        self.assertIn("system", actual_payload)
        self.assertEqual(actual_payload["system"], system_instructions)
        self.assertEqual(actual_payload['messages'][0]['content'], query_data["prompt"])

        expected_result = {"status": "success", "content": mock_response_text, "finish_reason": "end_turn", "usage": None}
        self.assertEqual(result, expected_result)

    def test_process_query_api_error(self):
        error_message = "Claude API unavailable"
        self.mock_api_manager.set_make_request_response(
            response_data={"error": "APIError", "message": error_message, "status_code": 503}
        )

        query_data = {"prompt": "test query for API error"}
        result = self.agent.process_query(query_data)
        
        self.mock_api_manager.make_request.assert_called_once()
        expected_result = {
            "status": "error",
            "message": f"API request failed: {error_message}",
            "details": {"error": "APIError", "message": error_message, "status_code": 503}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.error.assert_any_call(f"API request failed for Claude: {error_message}")

    def test_process_query_missing_prompt(self):
        query_data = {} # Empty query_data
        result = self.agent.process_query(query_data)
        
        expected_result = {"status": "error", "message": "User prompt missing"}
        self.assertEqual(result, expected_result)
        self.mock_api_manager.make_request.assert_not_called()
        self.agent.logger.error.assert_any_call("User prompt is missing in query_data.")

    def test_process_query_with_config_overrides(self):
        # Agent initialized with model="claude-test-model", temp=0.5, max_tokens=1000 in setUp
        
        mock_response_text = "Custom config response"
        mock_api_response = {"content": [{"type": "text", "text": mock_response_text}], "stop_reason": "max_tokens"}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {
            "prompt": "custom config test query",
            "temperature": 0.99, # Query-time override
            "max_tokens": 150     # Query-time override (Claude agent maps this to max_tokens_to_sample)
        }
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']

        self.assertEqual(actual_payload['model'], self.agent_config['model']) # Model from agent's init config
        self.assertEqual(actual_payload['temperature'], 0.99) # Overridden by query_data
        self.assertEqual(actual_payload['max_tokens'], 150)    # Overridden by query_data (used as max_tokens_to_sample)
        self.assertNotIn("system", actual_payload) # Default system prompt from config should be used if not overridden

        expected_result = {
            "status": "success", 
            "content": mock_response_text,
            "finish_reason": "max_tokens",
            "usage": None
        }
        self.assertEqual(result, expected_result)

    def test_process_query_invalid_api_response_structure_no_content_list(self):
        malformed_api_response = {"text_instead_of_content_list": "some_data"} # Missing 'content' list
        self.mock_api_manager.set_make_request_response(malformed_api_response)

        query_data = {"prompt": "test query for malformed response"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid response structure from Claude API (no content)", result["message"])
        self.agent.logger.error.assert_any_call(f"Unexpected response structure from Claude (no content list). Response: {malformed_api_response}")

    def test_process_query_invalid_api_response_structure_no_text_in_content(self):
        malformed_api_response = {"content": [{"type": "not_text", "data": "something"}]} # No 'text' field in content block
        self.mock_api_manager.set_make_request_response(malformed_api_response)

        query_data = {"prompt": "test query for malformed content block"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        self.assertEqual(result["status"], "error")
        self.assertIn("No text content found in Claude response", result["message"])
        self.agent.logger.error.assert_any_call(f"No text found in Claude response content blocks. Response: {malformed_api_response}")


if __name__ == '__main__':
    unittest.main()
