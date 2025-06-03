import unittest
from unittest.mock import MagicMock, AsyncMock # Added AsyncMock

# Assuming tests are run from the 'multi-agent-system' directory root
try:
    from src.agents.cursor_agent import CursorAgent
    from tests.mocks.mock_api_manager import MockAPIManager
except ImportError:
    # Fallback for different execution contexts
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.agents.cursor_agent import CursorAgent
    from tests.mocks.mock_api_manager import MockAPIManager

class TestCursorAgent(unittest.IsolatedAsyncioTestCase): # Changed inheritance
    def setUp(self):
        self.mock_api_manager = MockAPIManager() # MockAPIManager.make_request is now AsyncMock
        self.agent_config = {
            "model": "cursor-test-model",
            "mode": "code-gen-test",
            "default_system_prompt": "System for Cursor",
            "max_tokens": 1500, # Default from agent implementation (or typical)
            "temperature": 0.2  # Agent config default temperature
        }
        self.agent = CursorAgent(
            agent_name="TestCursor",
            api_manager=self.mock_api_manager, # type: ignore
            config=self.agent_config
        )
        self.agent.logger = MagicMock() # Suppress logging
        self.mock_api_manager.reset_mocks() # Ensure clean mocks for each test

    def test_initialization(self):
        self.assertEqual(self.agent.get_name(), "TestCursor")
        self.assertEqual(self.agent.model_mode, "code-gen-test")
        self.assertEqual(self.agent.default_system_prompt, "System for Cursor")
        self.assertEqual(self.agent.model, "cursor-test-model")
        self.agent.logger.info.assert_any_call("CursorAgent 'TestCursor' initialized with mode 'code-gen-test' and model 'cursor-test-model'.")

    def test_get_capabilities(self):
        caps = self.agent.get_capabilities()
        self.assertIn("description", caps)
        self.assertIn("capabilities", caps)
        self.assertIsInstance(caps["capabilities"], list)
        self.assertTrue(len(caps["capabilities"]) > 0)

    async def test_process_query_success(self): # Async
        mock_response_content = "Generated code by Cursor"
        mock_api_response = {"response": mock_response_content, "id": "curs_xxx", "raw_response": mock_api_response} # raw_response added for assertion
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test cursor query"}
        result = await self.agent.process_query(query_data) # Await

        self.mock_api_manager.make_request.assert_awaited_once() # Awaited
        call_args = self.mock_api_manager.make_request.call_args
        self.assertEqual(call_args[1]['service_name'], 'cursor')
        self.assertEqual(call_args[1]['endpoint'], 'compose')
        self.assertEqual(call_args[1]['method'], 'POST')
        
        actual_payload = call_args[1]['data']
        expected_full_prompt = f"{self.agent_config['default_system_prompt']}\n\nUser Query:\n{query_data['prompt']}"
        self.assertEqual(actual_payload['prompt'], expected_full_prompt)
        self.assertEqual(actual_payload['mode'], self.agent_config['mode'])
        self.assertEqual(actual_payload['model'], self.agent_config['model'])
        self.assertEqual(actual_payload['max_tokens'], self.agent_config['max_tokens'])
        self.assertEqual(actual_payload['temperature'], self.agent_config['temperature'])

        expected_result = {
            "status": "success", 
            "content": mock_response_content,
            "raw_response": mock_api_response # agent includes this
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.info.assert_any_call(f"Successfully received and parsed response from Cursor for query: '{query_data['prompt'][:100]}...'")

    async def test_process_query_success_with_system_prompt_override(self): # Async
        mock_response_content = "Cursor response with override"
        mock_api_response = {"response": mock_response_content, "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        custom_system_prompt = "Override system for Cursor"
        query_data = {"prompt": "test cursor query", "system_prompt": custom_system_prompt}
        result = await self.agent.process_query(query_data) # Await

        self.mock_api_manager.make_request.assert_awaited_once() # Awaited
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']
        
        expected_full_prompt = f"{custom_system_prompt}\n\nUser Query:\n{query_data['prompt']}"
        self.assertEqual(actual_payload['prompt'], expected_full_prompt)
        
        expected_result = {"status": "success", "content": mock_response_content, "raw_response": mock_api_response}
        self.assertEqual(result, expected_result)

    async def test_process_query_api_error(self): # Async
        error_message = "Cursor API is down"
        self.mock_api_manager.set_make_request_response(
            response_data={"error": "ServiceUnavailable", "message": error_message, "status_code": 503}
        )

        query_data = {"prompt": "test query for error"}
        result = await self.agent.process_query(query_data) # Await
        
        self.mock_api_manager.make_request.assert_awaited_once() # Awaited
        expected_result = {
            "status": "error",
            "message": f"API request failed: {error_message}",
            "details": {"error": "ServiceUnavailable", "message": error_message, "status_code": 503}
        }
        self.assertEqual(result, expected_result)
        self.agent.logger.error.assert_any_call(f"API request failed for Cursor: {error_message}")

    async def test_process_query_missing_prompt(self): # Async
        query_data = {} 
        result = await self.agent.process_query(query_data) # Await
        
        expected_result = {"status": "error", "message": "User query/prompt missing"}
        self.assertEqual(result, expected_result)
        self.mock_api_manager.make_request.assert_not_called() # Stays same
        self.agent.logger.error.assert_any_call("User query/prompt is missing in query_data.")

    async def test_process_query_response_parsing_generated_code_field(self): # Async
        mock_response_content = "Alternative code by Cursor"
        mock_api_response = {"generated_code": mock_response_content, "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test for generated_code field"}
        result = await self.agent.process_query(query_data) # Await
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], mock_response_content)

    async def test_process_query_response_parsing_text_field(self): # Async
        mock_response_content = "Text field content by Cursor"
        mock_api_response = {"text": mock_response_content, "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test for text field"}
        result = await self.agent.process_query(query_data) # Await
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], mock_response_content)

    async def test_process_query_response_parsing_completion_field(self): # Async
        mock_response_content = "Completion field content by Cursor"
        mock_api_response = {"completion": mock_response_content, "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)
        
        query_data = {"prompt": "test for completion field"}
        result = await self.agent.process_query(query_data) # Await
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["content"], mock_response_content)


    async def test_process_query_config_overrides_in_payload(self): # Async
        mock_api_response = {"response": "Config override response", "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {
            "prompt": "query for config overrides", 
            "max_tokens": 600, 
            "temperature_override": 0.88 
        }
        await self.agent.process_query(query_data) # Await

        self.mock_api_manager.make_request.assert_awaited_once() # Awaited
        actual_payload = self.mock_api_manager.make_request.call_args[1]['data']

        self.assertEqual(actual_payload['max_tokens'], 600) 
        self.assertEqual(actual_payload['temperature'], 0.88) 

    async def test_process_query_no_content_field_in_response(self): # Async
        mock_api_response = {"metadata": "some other data but no main content field", "raw_response": mock_api_response}
        self.mock_api_manager.set_make_request_response(mock_api_response)

        query_data = {"prompt": "query for no content field"}
        result = await self.agent.process_query(query_data) # Await

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "Invalid response structure from Cursor API.")
        self.agent.logger.error.assert_any_call(f"Failed to extract content from Cursor response. Response: {mock_api_response}")

if __name__ == '__main__':
    unittest.main()
