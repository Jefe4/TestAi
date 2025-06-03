import unittest
from unittest.mock import MagicMock

# Assuming tests are run from the 'multi-agent-system' directory root
try:
    from src.agents.windsurf_agent import WindsurfAgent
    from tests.mocks.mock_api_manager import MockAPIManager
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.windsurf_agent import WindsurfAgent
    from tests.mocks.mock_api_manager import MockAPIManager

class TestWindsurfAgent(unittest.TestCase):
    def setUp(self):
        self.mock_api_manager = MockAPIManager()
        self.agent_config = {
            "model": "windsurf-test-v1",
            "focus": "web-test-focus",
            "default_system_prompt": "System for Windsurf Test",
            "max_tokens": 1200,
            "temperature": 0.6
        }
        self.agent = WindsurfAgent(
            agent_name="TestWindsurf",
            api_manager=self.mock_api_manager, # type: ignore
            config=self.agent_config
        )
        self.agent.logger = MagicMock() # Suppress logging
        self.mock_api_manager.reset_mocks() # Ensure clean mocks

    def test_initialization(self):
        self.assertEqual(self.agent.get_name(), "TestWindsurf")
        self.assertEqual(self.agent.focus_area, "web-test-focus")
        self.assertEqual(self.agent.default_system_prompt, "System for Windsurf Test")
        self.assertEqual(self.agent.model, "windsurf-test-v1")
        self.agent.logger.info.assert_any_call(
            "WindsurfAgent 'TestWindsurf' initialized with focus 'web-test-focus' and model 'windsurf-test-v1'."
        )

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("description", caps)
        self.assertIn("capabilities", caps)
        self.assertIsInstance(caps["capabilities"], list)
        self.assertTrue(len(caps["capabilities"]) > 0)

    def test_process_query_success(self):
        mock_response_content = "Generated web content by Windsurf"
        mock_api_response = {"response": mock_response_content, "id": "wind_xyz"}
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test windsurf query"}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        call_args = self.mock_api_manager.make_request.call_args
        self.assertEqual(call_args[1]['service_name'], 'windsurf')
        self.assertEqual(call_args[1]['endpoint'], 'generate')
        self.assertEqual(call_args[1]['method'], 'POST')
        
        actual_payload = call_args[1]['data']
        expected_full_prompt = (f"System Prompt:\n{self.agent_config['default_system_prompt']}\n\n"
                                f"User Query:\n{query_data['prompt']}")
        self.assertEqual(actual_payload['prompt'], expected_full_prompt)
        self.assertEqual(actual_payload['focus'], self.agent_config['focus'])
        self.assertEqual(actual_payload['model'], self.agent_config['model'])
        self.assertEqual(actual_payload['max_tokens'], self.agent_config['max_tokens'])
        self.assertEqual(actual_payload['temperature'], self.agent_config['temperature'])

        expected_result = {
            "status": "success", 
            "content": mock_response_content,
            "raw_response": mock_api_response
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.info.assert_any_call(
            f"Successfully received and parsed response from Windsurf for query: '{query_data['prompt'][:100]}...'"
        )

    def test_process_query_success_with_system_prompt_override(self):
        mock_response_content = "Windsurf response with custom system prompt"
        mock_api_response = {"response": mock_response_content}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        custom_system_prompt = "Override system for Windsurf"
        query_data = {"prompt": "test windsurf query", "system_prompt": custom_system_prompt}
        result = self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']
        
        expected_full_prompt = (f"System Prompt:\n{custom_system_prompt}\n\n"
                                f"User Query:\n{query_data['prompt']}")
        self.assertEqual(actual_payload['prompt'], expected_full_prompt)
        
        expected_result = {"status": "success", "content": mock_response_content, "raw_response": mock_api_response}
        self.assertEqual(result, expected_result)

    def test_process_query_focus_override_from_query_data(self):
        mock_api_response = {"response": "Focus override response"}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        new_focus = "new-focus-area-for-windsurf"
        query_data = {"prompt": "test query with new focus", "focus": new_focus}
        self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']
        self.assertEqual(actual_payload['focus'], new_focus)

    def test_process_query_api_error(self):
        error_message = "Windsurf API service is currently down."
        self.mock_api_manager.set_make_request_response(
            response_data={"error": "ServiceFailure", "message": error_message, "status_code": 500}
        )

        query_data = {"prompt": "test query for Windsurf error"}
        result = self.agent.process_query(query_data)
        
        self.mock_api_manager.make_request.assert_called_once()
        expected_result = {
            "status": "error",
            "message": f"API request failed: {error_message}",
            "details": {"error": "ServiceFailure", "message": error_message, "status_code": 500}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.error.assert_any_call(f"API request failed for Windsurf: {error_message}")

    def test_process_query_missing_prompt(self):
        query_data = {} # Empty query
        result = self.agent.process_query(query_data)
        
        expected_result = {"status": "error", "message": "User query/prompt missing"}
        self.assertEqual(result, expected_result)
        self.mock_api_manager.make_request.assert_not_called()
        self.agent.logger.error.assert_any_call("User query/prompt is missing in query_data.")

    def test_process_query_response_parsing_generated_content_field(self):
        mock_response_content = "Alternative web content from Windsurf"
        mock_api_response = {"generated_content": mock_response_content} # Using 'generated_content'
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test for generated_content field in Windsurf"}
        result = self.agent.process_query(query_data)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], mock_response_content)

    def test_process_query_response_parsing_text_field(self):
        mock_response_content = "Text field content from Windsurf"
        mock_api_response = {"text": mock_response_content} # Using 'text'
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test for text field in Windsurf"}
        result = self.agent.process_query(query_data)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], mock_response_content)


    def test_process_query_config_overrides_in_payload(self):
        # setUp config: "max_tokens": 1200, "temperature": 0.6
        mock_api_response = {"response": "Config override response for Windsurf"}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {
            "prompt": "query for Windsurf config overrides", 
            "max_tokens": 700, # Override max_tokens
            "temperature_override": 0.11 # Override temperature
        }
        self.agent.process_query(query_data)

        self.mock_api_manager.make_request.assert_called_once()
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']

        self.assertEqual(actual_payload['max_tokens'], 700) # From query_data
        self.assertEqual(actual_payload['temperature'], 0.11) # From query_data "temperature_override"

    def test_process_query_no_content_field_in_response(self):
        mock_api_response = {"info": "Request processed, but no standard content field."}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {"prompt": "query for no content field in Windsurf"}
        result = self.agent.process_query(query_data)

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Invalid response structure from Windsurf API.")
        self.agent.logger.error.assert_any_call(
            f"Failed to extract content from Windsurf response. Response: {mock_api_response}"
        )

if __name__ == '__main__':
    unittest.main()
