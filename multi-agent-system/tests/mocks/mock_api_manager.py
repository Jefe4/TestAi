import unittest.mock
from unittest.mock import AsyncMock # Import AsyncMock

class MockAPIManager:
    def __init__(self):
        self.make_request = AsyncMock() # Changed to AsyncMock
        # Add any other methods from APIManager that agents might call, if any.
        # For now, make_request is the primary one.
        # If APIManager has other async methods that are called, they should also be AsyncMock.
        # Example: self.another_async_method = AsyncMock()

    def set_make_request_response(self, response_data, is_error=False, error_content=None):
        if is_error:
            # Default error structure from APIManager.make_request
            error_response = {
                "status": "error", # This key is not standard in APIManager error returns, 'error' is usually the key for the type
                "error": response_data, # e.g. "HTTPError", "Timeout"
                "message": error_content if error_content else "Simulated error", 
                "status_code": 500 # Default status code for simulated error
            }
            if isinstance(response_data, dict) and "status_code" in response_data : # if a full error dict is passed
                 self.make_request.return_value = response_data
            else: # construct from parts
                 self.make_request.return_value = error_response

            # If APIManager is expected to raise exceptions on certain errors:
            # self.make_request.side_effect = SomeExpectedException(response_data)
        else:
            self.make_request.return_value = response_data
    
    def reset_mocks(self):
        self.make_request.reset_mock()

# Example of how it might be used (optional to include in the file itself)
if __name__ == '__main__':
    mock_api = MockAPIManager()
    
    # Test setting a success response
    # Note: To test an AsyncMock, you'd typically await it in an async test function.
    # This __main__ block is synchronous, so direct assertions on awaitables are tricky.
    # We'll assume the set_make_request_response works as intended for AsyncMock's return_value.
    
    # Example (conceptual, as this __main__ is not async):
    # async def main_test():
    #     mock_api = MockAPIManager()
    #     mock_api.set_make_request_response({"data": "success_payload", "status_code": 200})
    #     success_result = await mock_api.make_request("test_service", "test_endpoint", method="GET")
    #     print(f"Success call result: {success_result}")
    #     mock_api.make_request.assert_awaited_once_with("test_service", "test_endpoint", method="GET")
    #     # ... other tests
    # if __name__ == '__main__':
    #     import asyncio
    #     asyncio.run(main_test())
    # For simplicity, the __main__ block will just show setup. Actual test is in test files.
    pass # Keep __main__ minimal or remove for a pure mock module.
    # print("MockAPIManager (now with AsyncMock make_request) basic setup shown.")
