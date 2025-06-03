import unittest
from unittest.mock import patch, MagicMock, AsyncMock # Use AsyncMock for async methods
import asyncio
import aiohttp # For type hinting and specific exceptions

try:
    from src.utils.api_manager import APIManager
    from src.utils.logger import get_logger # APIManager imports this
except ImportError:
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.utils.api_manager import APIManager
    from src.utils.logger import get_logger


class TestAPIManager(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Patch get_logger before APIManager is instantiated
        self.logger_patcher = patch('src.utils.api_manager.get_logger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger_instance = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger_instance

        # Mock for the response object that session.request() context manager returns
        self.mock_response_instance = MagicMock(spec=aiohttp.ClientResponse)
        self.mock_response_instance.json = AsyncMock(return_value={"key": "value"})
        self.mock_response_instance.text = AsyncMock(return_value="some text")
        self.mock_response_instance.status = 200
        self.mock_response_instance.reason = "OK"
        self.mock_response_instance.headers = {'Content-Type': 'application/json'} # CaseInsensitiveMultiDict
        # For aiohttp response.content_type, it's an attribute
        type(self.mock_response_instance).content_type = PropertyMock(return_value='application/json')


        # Mock for the context manager returned by session.request()
        self.mock_response_cm = AsyncMock() # This mock will be the one returned by session.request()
        self.mock_response_cm.__aenter__.return_value = self.mock_response_instance
        self.mock_response_cm.__aexit__.return_value = None # Or mock it to do something if needed

        # Mock for the session instance itself
        self.mock_session_instance = MagicMock(spec=aiohttp.ClientSession)
        self.mock_session_instance.request = MagicMock(return_value=self.mock_response_cm)
        self.mock_session_instance.closed = False
        self.mock_session_instance.close = AsyncMock() # For testing close_session

        # Patch aiohttp.ClientSession to return our mock_session_instance
        self.aiohttp_session_patcher = patch('src.utils.api_manager.aiohttp.ClientSession', return_value=self.mock_session_instance)
        self.mock_aiohttp_client_session_class = self.aiohttp_session_patcher.start()
        
        self.addCleanup(self.aiohttp_session_patcher.stop)
        self.addCleanup(self.logger_patcher.stop)

        # Instantiate APIManager *after* critical patches are set up
        self.api_manager = APIManager(config_path=None) # config_path=None uses default
        
        # Further mock parts of APIManager that interact with file system or env for unit test isolation
        self.api_manager.load_service_configs = MagicMock() # Prevent actual file loading
        self.api_manager.service_configs = {
            "test_service": {"api_key": "test_key_123", "base_url": "http://fakeapi.com/api"},
            "service_no_key": {"base_url": "http://anotherapi.com/api"},
            "claude_service": {"api_key": "claude_key", "base_url": "http://claude.com/api"}
        }
        # Reset logger mock for APIManager instance specific logs after init
        self.mock_logger_instance.reset_mock()


    async def test_get_session_creates_and_reuses_session(self):
        self.assertIsNone(self.api_manager.client_session)
        
        session1 = await self.api_manager._get_session()
        self.assertIsNotNone(session1)
        self.mock_aiohttp_client_session_class.assert_called_once() # Check ClientSession was instantiated
        self.assertIs(self.api_manager.client_session, session1)

        session2 = await self.api_manager._get_session()
        self.assertIs(session1, session2) # Should reuse existing session
        self.mock_aiohttp_client_session_class.assert_called_once() # Still called only once

        # Simulate closed session
        self.api_manager.client_session.closed = True
        session3 = await self.api_manager._get_session()
        self.assertIsNotNone(session3)
        self.assertIsNot(session1, session3) # Should be a new session
        self.assertEqual(self.mock_aiohttp_client_session_class.call_count, 2) # Called again


    async def test_make_request_success_json_response(self):
        self.mock_response_instance.status = 200
        type(self.mock_response_instance).content_type = PropertyMock(return_value='application/json')
        self.mock_response_instance.json = AsyncMock(return_value={"data": "success"})
        
        response = await self.api_manager.make_request("test_service", "test_endpoint", method="POST", data={"req": "data"})
        
        self.mock_session_instance.request.assert_called_once()
        args, kwargs = self.mock_session_instance.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "http://fakeapi.com/api/test_endpoint")
        self.assertIn("json", kwargs)
        self.assertEqual(kwargs["json"], {"req": "data"})
        self.assertEqual(response, {"data": "success", "status": "success"}) # APIManager adds status if not present

    async def test_make_request_success_text_response(self):
        self.mock_response_instance.status = 200
        type(self.mock_response_instance).content_type = PropertyMock(return_value='text/plain')
        self.mock_response_instance.text = AsyncMock(return_value="OK text")

        response = await self.api_manager.make_request("test_service", "test_endpoint_text", method="GET")
        
        self.assertEqual(response, {"status": "success", "content": "OK text", "status_code": 200})

    async def test_make_request_http_error(self):
        self.mock_response_instance.status = 500
        self.mock_response_instance.reason = "Server Error"
        type(self.mock_response_instance).content_type = PropertyMock(return_value='application/json')
        error_payload = {"error_detail": "detail"}
        self.mock_response_instance.json = AsyncMock(return_value=error_payload) # Error response might be JSON

        response = await self.api_manager.make_request("test_service", "error_endpoint")
        
        self.assertEqual(response["status"], "error")
        self.assertIn("API request failed with status 500", response["message"])
        self.assertIn(str(error_payload), response["message"])
        self.assertEqual(response["status_code"], 500)

    async def test_make_request_client_connector_error(self):
        # request() itself raises this, not the response object
        self.mock_session_instance.request.side_effect = aiohttp.ClientConnectorError(MagicMock(spec=aiohttp.ClientRequestInfo), OSError("connection error"))
        
        response = await self.api_manager.make_request("test_service", "conn_error_endpoint")
        
        self.assertEqual(response["status"], "error")
        self.assertIn("AIOHTTP ClientConnectorError", response["message"])
        self.assertEqual(response["status_code"], 503)

    async def test_make_request_generic_client_error(self):
        self.mock_session_instance.request.side_effect = aiohttp.ClientError("Some generic client error")
        response = await self.api_manager.make_request("test_service", "generic_client_error_endpoint")
        self.assertEqual(response["status"], "error")
        self.assertIn("AIOHTTP ClientError: Some generic client error", response["message"])
        self.assertEqual(response["status_code"], 500)


    async def test_make_request_timeout_error(self):
        self.mock_session_instance.request.side_effect = asyncio.TimeoutError()
        response = await self.api_manager.make_request("test_service", "timeout_endpoint")
        self.assertEqual(response, {"status": "error", "message": "Request timed out", "status_code": 408})

    async def test_make_request_unexpected_error(self):
        self.mock_session_instance.request.side_effect = ValueError("Unexpected issue")
        response = await self.api_manager.make_request("test_service", "unexpected_endpoint")
        self.assertEqual(response["status"], "error")
        self.assertIn("Unexpected error: Unexpected issue", response["message"])
        self.assertEqual(response["status_code"], 500)

    async def test_close_session_closes_active_session(self):
        # Ensure session is "created"
        await self.api_manager._get_session() 
        self.assertIsNotNone(self.api_manager.client_session)
        self.api_manager.client_session.closed = False # Explicitly set for clarity
        
        await self.api_manager.close_session()
        self.mock_session_instance.close.assert_awaited_once()
        self.assertIsNone(self.api_manager.client_session) # Session should be set to None after close

    async def test_close_session_no_active_session(self):
        self.assertIsNone(self.api_manager.client_session)
        await self.api_manager.close_session()
        # mock_session_instance.close should not have been called if session was None
        self.mock_session_instance.close.assert_not_called()

    async def test_close_session_already_closed(self):
        await self.api_manager._get_session()
        self.api_manager.client_session.closed = True # Simulate already closed
        await self.api_manager.close_session()
        self.mock_session_instance.close.assert_not_called() # Should not call close on already closed

    async def test_make_request_get_method_with_data_uses_params(self):
        await self.api_manager.make_request("test_service", "get_endpoint", method="GET", data={"key": "val"})
        _, kwargs = self.mock_session_instance.request.call_args
        self.assertIn("params", kwargs)
        self.assertEqual(kwargs["params"], {"key": "val"})
        self.assertNotIn("json", kwargs)

    async def test_make_request_get_method_with_explicit_params(self):
        await self.api_manager.make_request("test_service", "get_endpoint", method="GET", params={"p_key": "p_val"})
        _, kwargs = self.mock_session_instance.request.call_args
        self.assertIn("params", kwargs)
        self.assertEqual(kwargs["params"], {"p_key": "p_val"})

    async def test_make_request_post_method_with_explicit_params_and_data(self):
        # POST with JSON data and also query parameters
        await self.api_manager.make_request("test_service", "post_endpoint", method="POST", 
                                            data={"body_key": "body_val"}, 
                                            params={"q_param": "q_val"})
        _, kwargs = self.mock_session_instance.request.call_args
        self.assertIn("json", kwargs)
        self.assertEqual(kwargs["json"], {"body_key": "body_val"})
        self.assertIn("params", kwargs)
        self.assertEqual(kwargs["params"], {"q_param": "q_val"})
        
    async def test_auth_header_for_claude_includes_content_type(self):
        # Claude's auth header includes "Content-Type": "application/json"
        # make_request also defaults to this. Test if it's correctly set.
        self.mock_response_instance.status = 200
        type(self.mock_response_instance).content_type = PropertyMock(return_value='application/json')
        self.mock_response_instance.json = AsyncMock(return_value={"data": "success"})

        await self.api_manager.make_request("claude_service", "claude_endpoint", method="POST", data={"req": "data"})
        _, kwargs = self.mock_session_instance.request.call_args
        self.assertIn("headers", kwargs)
        # The auth_header for Claude provides Content-Type, this should be in final_headers
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertIn("x-api-key", kwargs["headers"])


if __name__ == '__main__':
    unittest.main()
